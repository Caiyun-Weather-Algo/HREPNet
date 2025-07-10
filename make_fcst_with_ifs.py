import os
import time
import arrow
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
import argparse
import logging
project_path = os.path.dirname(os.path.abspath(__file__))
import sys 
sys.path.append(project_path)

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from src.models.build_model import build_model
from src.utils.data_utils import dbz2tp
from src.utils.plot_util import plot_pred_tp
from src.datasets.build_loader import get_dataloader, get_dataset
from src.utils.dist import setup, cleanup
from src.utils.logger import create_logger
 

def test_loop(rank, model, ema, vae_model, diffusion, test_loader, cfg, logger):
    model.eval()
    ema.eval()
   
    en = 0.94 * 42.40353617054452
    cn = 46.29922575990931
    for i, data in enumerate(test_loader):
        init_timestamp, fcst_timestamp, input, input0 = data  
        # dtm 
        dtm_input = [input[0].to(rank), input[1].to(rank)]  
        with torch.no_grad():               
            output = model(dtm_input) 
        # clear memory
        del dtm_input   
        input = None  
        output[output < 2] = 2
        # prob
        prob_input = torch.cat((input0[0].to(rank), output/en, input0[1].to(rank)), dim=1)
        model_kwargs = dict(cond=prob_input[..., 0:180, 0:280])
        sample_fn = ema.forward
        # Sample images
        n = output.shape[0]
        ens_members = []
        era5_pred_5km_dbz = torch.nn.functional.interpolate(output[..., 0:180, 0:280], scale_factor=5, mode='nearest')
        
        for m in range(cfg.member_number):
            t0 = time.time()
            if torch.cuda.device_count() > 1:
                z = torch.randn(n, ema.module.in_channels, ema.module.img_size[0], ema.module.img_size[1], device=rank)
            else:
                z = torch.randn(n, ema.in_channels, ema.img_size[0], ema.img_size[1], device=rank)
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=rank) 
            if rank==0:
                logger.info(f"Batch {i}: Sampling {m} images costs: {time.time() - t0}s on rank {rank}")
            # VAE decoding
            residual = vae_model.decode(samples / cfg.scale_factor)
            pred_dbz = (residual + era5_pred_5km_dbz/en) * cn
            pred_tp = dbz2tp(pred_dbz.cpu().detach().numpy())
            ens_members.append(pred_tp)

        ens_members = np.array(ens_members)
        dtm_tp = dbz2tp(output.cpu().numpy()) 
        
        init_times = [arrow.get(int(tm.cpu().numpy())).format('YYYYMMDDHH') for tm in init_timestamp]
        fcst_times = [arrow.get(int(tm.cpu().numpy())).format('YYYYMMDDHH') for tm in fcst_timestamp]
        t = 0
        t0 = time.time()
        for init_time, fcst_time in zip(init_times, fcst_times):
            img_path = f'{cfg.result_path}/images/{init_time}'
            os.makedirs(img_path, exist_ok=True)
            plot_pred_tp(f"{img_path}/tp_cmpa_{init_time}_{fcst_time}", 
                                "rainfall_cmpa",
                                init_time,
                                fcst_time,
                                dtm_tp[t, 0, ::-1],
                                ens_members[:, t, 0, ::-1], 
                                region='cmpa',
                                )
         
            # store data 
            data_path = f'{cfg.result_path}/data/{init_time}'
            os.makedirs(data_path, exist_ok=True)
            np.savez(f'{data_path}/dtm_{init_time}_{fcst_time}.npz', dtm=dtm_tp[t])
            np.savez(f'{data_path}/ens_{init_time}_{fcst_time}.npz', ens=ens_members[:, t])
            t += 1
        if rank==0:
            logger.info(f'Save images at {img_path}, result at {data_path}, cost {time.time() - t0}s')


def run_eval(rank, world_size, cfg, log_filename):
    device = torch.device("cuda", rank)
    if cfg.model.mp:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        # log
        logger = create_logger(output_dir=cfg.result_path, dist_rank=dist.get_rank(), name=log_filename)  
        if dist.get_rank() == 0:        
            logger.info(f"Training configs: {OmegaConf.to_yaml(cfg)}")

        seed = 42 * dist.get_world_size() + rank
    else:
        logger = create_logger(output_dir=cfg.result_path, dist_rank=0, name=log_filename)  
        seed = 42

    torch.manual_seed(seed)
     
    # DTM model
    model = build_model(cfg.dtm_model, cfg)     
    model = model.to(device)  
    ckpt = torch.load(f"{cfg.model_path}/{cfg.dtm_model}_epoch30.pth.tar", map_location="cuda:0") 
    model.load_state_dict(ckpt["model_state_dict"])
    # model DDP
    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Prob model
    prob_model = build_model(cfg.prob_model, cfg)  
    prob_model = prob_model.to(device)
    ckpt = torch.load(f"{cfg.model_path}/{cfg.prob_model}_epoch90.pth.tar", map_location="cpu")
    prob_model.load_state_dict(ckpt["ema_state_dict"])   
    # model DDP
    if torch.cuda.device_count() > 1:
        prob_model = DDP(prob_model, device_ids=[rank], output_device=rank)                   
    
    # load vae model 
    vae_name = cfg.model[cfg.prob_model]['vaeconfig']['vae_model']
    vae_model = build_model(vae_name, cfg)
    vae_ckpt = torch.load(f"{cfg.model_path}/{vae_name}_best.pth.tar", map_location="cpu")
    vae_model.load_state_dict(vae_ckpt['model_state_dict'])
    vae_model = vae_model.to(device)
    
    # diffusion
    cfg.model[cfg.prob_model]['diffconfig'].update({"timestep_respacing": cfg.model[cfg.prob_model]['diffconfig']["num_sampling_steps"]})
    diffusion_sampling = build_model("diffusion", cfg)
    
    # dataset
    era5_test_dataset = get_dataset(mode="test", cfg=cfg)
    # data loader
    test_loader = get_dataloader(era5_test_dataset, cfg.hyper_params.batch_size, mode="test", flag_ddp=cfg.model.mp)
    
    # test
    test_loop(rank, model, prob_model, vae_model, diffusion_sampling, test_loader, cfg, logger)
    
    cleanup()


def predict(start_time: str, end_time: str, model_path: str, result_path: str, ec_path: str, log_filename):
    cfg = OmegaConf.load(f"{project_path}/config/predict_config.yaml")
    cfg.update({
        "start_time": start_time,
        "end_time": end_time,
        "result_path": result_path,
        "ec_path": ec_path,
        "model_path": model_path
    })
    if cfg.model.mp:
        world_size = 2
        mp.spawn(run_eval,
                args=(world_size, cfg, log_filename),
                nprocs=2,
                join=True)
    else:
        run_eval(rank=0, world_size=1, cfg=cfg, log_filename=log_filename)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict time range.")
    parser.add_argument('--start_time', type=str, required=True, help='Start time of the prediction') #yyyy-mm--dd_hh
    parser.add_argument('--end_time', type=str, required=True, help='End time of the prediction') #yyyy-mm--dd_hh
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--result_path', type=str, required=True, help='Path to the result')
    parser.add_argument('--ec_path', type=str, required=True, help='Path to the EC data')
    parser.add_argument('--log_filename', type=str, required=True, help='Log filename')
    args = parser.parse_args()
    
    predict(args.start_time, args.end_time, args.model_path, args.result_path, args.ec_path, args.log_filename)
    # python make_fcst_with_ifs.py --start_time 2025-07-09_12 --end_time 2025-07-09_12 --ec_path '/home/hess/bucket/hess/datasets/hres' --result_path './results' --model_path './model_weights' --log_filename 'fcst_with_ifs'