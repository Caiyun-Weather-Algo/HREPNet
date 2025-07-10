import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import time
import arrow
import yaml
import hydra
import numpy as np 
from typing import OrderedDict
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.models.build_model import build_model
from src.utils.plot_util import plot_train_result
from src.utils.dist import setup, cleanup
from src.datasets.prob_dataset import ProbDataset

variables = OrderedDict(yaml.load(open("./config/era5_variables.yaml", "r"), Loader=yaml.Loader))  
 
           
def test_one_epoch(rank, ema, vae_model, diffusion, test_loader, img_base_path, cfg): 
    ema.eval()
    
    # prepare plot 
    member_number = 11
    step = cfg.model[cfg.model.name]['diffconfig']["num_sampling_steps"]
    img_path = img_base_path+f'/m{member_number}_mean_step{step}'
    os.makedirs(img_path, exist_ok=True)
    data_path = f'./result/m{member_number}_mean_step{step}_ep90'
    os.makedirs(data_path, exist_ok=True) 
        
    for i, data in enumerate(test_loader):
        timestamp, x_surf, x_high, era5_tp, cmpa_tp = data
        timestamp = timestamp.detach().cpu().numpy()
        ymdh = arrow.get(int(timestamp[0])).format('YYYY-MM-DD_HH')
        np.savez(f'{data_path}/tp_{ymdh}.npz', 
                 cmpa=cmpa_tp.cpu().numpy(), 
                 era5_tp=era5_tp.cpu().numpy(), 
                )

        # preprocessing
        x = torch.cat((x_surf, era5_tp, x_high), dim=1).to(rank)
        era5_tp = era5_tp.to(rank)
        cmpa_tp = cmpa_tp.to(rank)
        era5_tp = torch.nn.functional.interpolate(era5_tp[..., 0:180, 0:280], scale_factor=5, mode='nearest')
        y = cmpa_tp - era5_tp

        # encode 
        with torch.no_grad():
            # Map input images to latent space + normalize latents:      
            encoding = vae_model.encode(y).sample().mul_(cfg.scale_factor)  # 0.18215 is the scale factor of latent
  
        # forward       
        model_kwargs = dict(cond=x[..., 0:180, 0:280])
        sample_fn = ema.forward
       
        # Sample images
        n = x_surf.shape[0]
        for m in range(member_number):
            t0 = time.time()
            z = torch.randn(n, ema.module.in_channels, ema.module.img_size[0], ema.module.img_size[1], device=rank)
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=rank) 
            print(f"{ymdh}: sampling {m} images costs: {time.time() - t0}, {rank}")

            # VAE decoding
            residual = vae_model.decode(samples / cfg.scale_factor)
            np.savez(f'{data_path}/res_{ymdh}_m{m}.npz', residual=residual.detach().cpu().numpy())

            # continue
            pred = residual + era5_tp 

            if m == 0:
                plot_train_result(f"{img_path}/tp_residual_{ymdh}_{m}", 
                        "radar_diff_norm",
                        residual.detach().cpu().numpy()[0, 0, ::-1], 
                        y.detach().cpu().numpy()[0, 0, ::-1],
                        region='cmpa_5km')
            
                     
def ddp_func(rank, world_size, cfg):
    # path
    result_path = f"{cfg.base_path}/exps/{cfg.model.name}_exp{cfg.exp}"
    img_base_path = f"{result_path}/images"
    model_path = f"{result_path}/models"
    log_path = f"{result_path}/logs"
    os.makedirs(img_base_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # set up DDP
    if cfg.model.mp:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        seed = 42 * dist.get_world_size() + rank
    else:
        seed = 42
    device = torch.device("cuda", rank)
    torch.manual_seed(seed)
    
    # create model
    model = build_model(cfg.model.name, cfg)  
    
    # load checkpoint
    ckpt = torch.load("{}/{}_epoch90.pth.tar".format(model_path, cfg.model.name), map_location="cpu", weights_only=False)
    if ckpt is not None:
        model.load_state_dict(ckpt["ema_state_dict"])   
    # model DDP
    if torch.cuda.device_count() > 1:
        model = DDP(model.to(device), device_ids=[rank], output_device=rank)                   

    # load vae model 
    vae_name = cfg.model[cfg.model.name]['vaeconfig']['vae_model']
    vae_path = cfg.model[cfg.model.name]['vaeconfig']['vae_path']
    vae_model = build_model(vae_name, cfg)
    vae_ckpt = torch.load("{}/{}_best.pth.tar".format(vae_path, vae_name), map_location="cpu", weights_only=False)
    vae_model.load_state_dict(vae_ckpt['model_state_dict'])
    vae_model = vae_model.to(device)
    
    # diffusion
    cfg.model[cfg.model.name]['diffconfig'].update({"timestep_respacing": cfg.model[cfg.model.name]['diffconfig']["num_sampling_steps"]})
    diffusion_sampling = build_model("diffusion", cfg)
    
    # Dataloader
    era5_test_dataset = ProbDataset(
                start_time=cfg.start_time,
                end_time=cfg.end_time,
                valid_time=cfg.valid_time,
                test_time=cfg.test_time,
                input_vars=cfg.input,
                output_vars=cfg.output,
                mode="test",
                input_step=cfg.hyper_params.input_step,
                forecast_step=cfg.hyper_params.forecast_step, 
                start_lead=cfg.hyper_params.start_lead, 
                sample_interval=cfg.hyper_params.sample_interval,
                use_static=cfg.hyper_params.use_static, 
                add_latlon_time=cfg.hyper_params.add_latlon_time,
                norm_method=cfg.hyper_params.norm_method,
                era5_path=cfg.era5_path,
                cmpa_path=cfg.cmpa_path,
                )
    sampler = torch.utils.data.distributed.DistributedSampler(era5_test_dataset, shuffle=False)
    test_loader = DataLoader(era5_test_dataset,
                            batch_size=1, #cfg.hyper_params.batch_size,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=4, 
                            sampler=sampler, 
                            # multiprocessing_context="spawn", 
                            )       
    # test
    test_one_epoch(rank, model, vae_model, diffusion_sampling, test_loader, img_base_path, cfg)
    
    cleanup()
    return


@hydra.main(version_base=None, config_path="./config", config_name="train_tp_config.yaml")
def main(cfg):
    if cfg.model.mp:
        world_size = 2
        mp.spawn(ddp_func,
                args=(world_size, cfg),
                nprocs=2,
                join=True)  
    else:
        ddp_func(rank=0, world_size=1, cfg=cfg)


if __name__ == '__main__':
    main()
    
