import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import time
import arrow
import yaml
import hydra
import numpy as np 
from omegaconf import OmegaConf
from typing import OrderedDict
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from src.utils.data_utils import dbz2tp

from src.models.build_model import build_model
from src.utils.plot_util import plot_train_result, plot_result_womap
from src.datasets.prob_dataset import ProbDataset

variables = OrderedDict(yaml.load(open("./config/era5_variables.yaml", "r"), Loader=yaml.Loader))  
 
           
def test_one_epoch(rank, ema, vae_model, diffusion, test_loader, img_base_path, cfg): 
    ema.eval()
    cmpa_norm_scale = 46.29922575990931
    # prepare plot 
    member_number = 1  # 11
    step = cfg.model[cfg.model.name]['diffconfig']["num_sampling_steps"]
    img_path = img_base_path+f'/eval_m{member_number}_mean_step{step}'
    os.makedirs(img_path, exist_ok=True)
        
    for i, data in enumerate(test_loader):
        timestamp, x_surf, x_high, era5_tp, cmpa_tp = data
        
        timestamp = timestamp.detach().cpu().numpy()
        ymdh = arrow.get(int(timestamp[0])).format('YYYY-MM-DD_HH')
      
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
        t0 = time.time()
        n = x_surf.shape[0]
        for m in range(member_number):
            z = torch.randn(n, ema.in_channels, ema.img_size[0], ema.img_size[1], device=rank)
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=rank) 
            print(f"sampling images costs: {time.time() - t0}")

            # VAE decoding
            residual = vae_model.decode(samples / cfg.scale_factor)
            
            # total rainfall
            print(residual.min(), residual.max(), era5_tp.min(), era5_tp.max())
            pred = residual + era5_tp 
            
            # rainfall
            target = dbz2tp(cmpa_tp.detach().cpu().numpy() * cmpa_norm_scale)
            output = dbz2tp(pred.detach().cpu().numpy() * cmpa_norm_scale)
            plot_train_result(f"{img_path}/rain_cmpa_{ymdh}", 
                            "rainfall",
                            output[0, 0, ::-1], 
                            target[0, 0, ::-1],
                            region='cmpa_5km')
        
            plot_train_result(f"{img_path}/tp_residual_{ymdh}", 
                            "radar_diff_norm",
                            residual.detach().cpu().numpy()[0, 0, ::-1], 
                            y.detach().cpu().numpy()[0, 0, ::-1],
                            region='cmpa_5km')
            
            plot_train_result(f"{img_path}/tp_cmpa_{ymdh}", 
                            "radar",
                            pred.detach().cpu().numpy()[0, 0, ::-1] * cmpa_norm_scale, 
                            cmpa_tp.detach().cpu().numpy()[0, 0, ::-1] * cmpa_norm_scale,
                            region='cmpa_5km')
            
                     
def ddp_func(cfg):
    # path
    cfg.exp = "p7"
    cfg.model.name = "condDiT"
    result_path = f"{cfg.base_path}/exps/{cfg.model.name}_exp{cfg.exp}"
    img_base_path = f"{result_path}/images"
    model_path = f"{result_path}/models"
    log_path = f"{result_path}/logs"
    os.makedirs(img_base_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42 * 2)
    
    # create model
    model = build_model(cfg.model.name, cfg)  
    model = model.to(device)
    
    # load checkpoint
    ckpt = torch.load("{}/{}_epoch90.pth.tar".format(model_path, cfg.model.name), map_location="cpu", weights_only=False)
    if ckpt is not None:
        model.load_state_dict(ckpt["ema_state_dict"])   
    
    # for n,p in model.named_parameters():
    #     print(n, p)
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
                test_time= "2021-08-01_01", #cfg.test_time,
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
    test_loader = DataLoader(era5_test_dataset,
                            batch_size=1, #cfg.hyper_params.batch_size,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=0, 
                            sampler=None, 
                            # multiprocessing_context="spawn", 
                            )    
    
    # test
    test_one_epoch(device, model, vae_model, diffusion_sampling, test_loader, img_base_path, cfg)
    return


@hydra.main(version_base=None, config_path="./config", config_name="train_tp_config.yaml")
def main(cfg):
    ddp_func(cfg=cfg)


if __name__ == '__main__':
    main()
    
