import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import time
import arrow
import yaml
import hydra
from omegaconf import OmegaConf
from typing import OrderedDict
from copy import deepcopy
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

from src.utils.model_utils import get_optimizer, get_scheduler, load_checkpoint, save_diffusion_model, LOSSES
from src.models.build_model import build_model
from src.utils.plot_util import plot_train_result
from src.utils.dist import setup, cleanup
from src.utils.logger import create_logger
from src.datasets.build_loader import get_dataset, get_dataloader
from src.utils.training_helper import update_ema, requires_grad

variables = OrderedDict(yaml.load(open("./config/era5_variables.yaml", "r"), Loader=yaml.Loader))

     
def train_one_epoch(rank, ema, model, vae_model, diffusion, train_loader, loss_func, optimizer, scheduler, writer, cfg, epoch):
    iters = len(train_loader)
    train_loss = torch.tensor(0.0).to(rank)
   
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    for i, data in enumerate(train_loader): 
        print(f"epoch {epoch}, iter {i}")
        x_surf, x_high, era5_tp, cmpa_tp = data
        # preprocessing
        x = torch.cat((x_surf, era5_tp, x_high), dim=1).to(rank)
        y = cmpa_tp - torch.nn.functional.interpolate(era5_tp[..., 0:180, 0:280], scale_factor=5, mode='nearest')
        y = y.to(rank)
        
        with torch.no_grad():
            # Map input images to latent space + normalize latents:      
            z = vae_model.encode(y).sample().mul_(cfg.scale_factor)  # 0.18215 is the scale factor of latent
       
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],)).to(rank)
        model_kwargs = dict(cond=x[..., 0:180, 0:280])

        # train
        loss_dict = diffusion.training_losses(model, z, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(ema, model.module)

        if cfg.hyper_params.scheduler=="cosineannealingwarmrestart" or cfg.hyper_params.scheduler=="cosineannealing":
            scheduler.step(epoch + i / iters)
            
        train_loss += loss.detach()
    train_loss = train_loss / len(train_loader)
    return train_loss, scheduler  
 
           
def test_one_epoch(rank, ema, vae_model, diffusion, test_loader, loss_func, img_base_path, logger, cfg, epoch, iters): 
    ema.eval()
    test_loss = torch.tensor(0.0).to(rank)
    if epoch % 10 == 0:
        for i, data in enumerate(test_loader):
            timestamp, x_surf, x_high, era5_tp, cmpa_tp = data
            # preprocessing
            x = torch.cat((x_surf, era5_tp, x_high), dim=1).to(rank)
            era5_tp = era5_tp.to(rank)
            cmpa_tp = cmpa_tp.to(rank)

            era5_tp = torch.nn.functional.interpolate(era5_tp[..., 0:180, 0:280], scale_factor=5, mode='nearest')
            y = cmpa_tp - era5_tp

            # forward       
            model_kwargs = dict(cond=x[..., 0:180, 0:280])
            sample_fn = ema.forward

            # Sample images
            t0 = time.time()
            n = x_surf.shape[0]
            z = torch.randn(n, ema.in_channels, ema.img_size[0], ema.img_size[1], device=rank)
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=rank) 
            logger.info(f"sampling images costs: {time.time() - t0}")

            # VAE decoding
            resdiual = vae_model.decode(samples / cfg.scale_factor)
            pred = resdiual + era5_tp 
                
            # plot mid-results
            t0 = time.time()
            img_path = img_base_path+f'/epoch{epoch}'
            os.makedirs(img_path, exist_ok=True)
            if (cfg.model.mp and dist.get_rank()==0) or not cfg.model.mp:
                timestamp = timestamp.detach().cpu().numpy()
                ymdh = arrow.get(int(timestamp[0])).format('YYYY-MM-DD_HH')
                plot_train_result(f"{img_path}/tp_cmpa_{ymdh}", 
                                "radar",
                                pred.detach().cpu().numpy()[0, 0, ::-1] * 46.29922575990931, 
                                cmpa_tp.detach().cpu().numpy()[0, 0, ::-1] * 46.29922575990931,
                                region='cmpa_5km')
                
                plot_train_result(f"{img_path}/tp_residual_{ymdh}", 
                        "radar_diff_norm",
                        resdiual.detach().cpu().numpy()[0, 0, ::-1], 
                        y.detach().cpu().numpy()[0, 0, ::-1],
                        region='cmpa_5km')
                logger.info(f"saving images costs: {time.time() - t0}")
    test_loss = test_loss / len(test_loader) 
    return test_loss


def pre_processing(input, rank, model_name):
    if "unet" in model_name or 'swinir' in model_name:
        input = torch.cat((input[0], input[1]), dim=1)
        input = input.to(rank)
    elif "transformer" in model_name or 'afnonet' in model_name:
        input = [input[0].to(rank), input[1].to(rank)]  
    elif 'autoencoder' in model_name:
        era5_tp, cmpa_tp = input[0].to(rank), input[1].to(rank)
        input = cmpa_tp - torch.nn.functional.interpolate(era5_tp[..., 0:180, 0:280], scale_factor=5, mode='nearest')
    return input 
    
                     
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
        # log
        logger = create_logger(output_dir=result_path, dist_rank=dist.get_rank(), name=cfg.model.name)  
        if dist.get_rank() == 0:        
            logger.info(f"Training configs: {OmegaConf.to_yaml(cfg)}")
        seed = 42 * dist.get_world_size() + rank
    else:
        logger = create_logger(output_dir=result_path, dist_rank=0, name=cfg.model.name)  
        seed = 42
    torch.manual_seed(seed)
    device = torch.device("cuda", rank)
  
    writer = SummaryWriter(log_dir=log_path)
   
    # create model
    logger.info(f"Creating model {cfg.model.name}")
    model = build_model(cfg.model.name, cfg)  
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of model params: {n_parameters}")  
    
    # load checkpoint
    ckpt, start_epoch = load_checkpoint(cfg, model_path)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)      
    
    # model DDP
    if torch.cuda.device_count() > 1:
        model = DDP(model.to(device), device_ids=[rank], output_device=rank)                   

    # Prepare models for training:
    if torch.cuda.device_count() > 1:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    else:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    if ckpt is not None:
        ema.load_state_dict(ckpt["ema_state_dict"])
  
    # load vae model 
    vae_name = cfg.model[cfg.model.name]['vaeconfig']['vae_model']
    vae_path = cfg.model[cfg.model.name]['vaeconfig']['vae_path']
    vae_model = build_model(vae_name, cfg)
    vae_ckpt = torch.load("{}/{}_best.pth.tar".format(vae_path, vae_name), map_location="cpu")
    vae_model.load_state_dict(vae_ckpt['model_state_dict'])
    vae_model = vae_model.to(device)
    
    # diffusion
    diffusion = build_model("diffusion", cfg) 
    cfg.model[cfg.model.name]['diffconfig'].update({"timestep_respacing": cfg.model[cfg.model.name]['diffconfig']["num_sampling_steps"]})
    diffusion_sampling = build_model("diffusion", cfg)
    
    # optimizer and scheduler
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)  
    
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
       
    # Dataloader
    era5_train_dataset = get_dataset(mode="train", cfg=cfg)
    era5_test_dataset = get_dataset(mode="test", cfg=cfg)
    train_loader = get_dataloader(era5_train_dataset, cfg.hyper_params.batch_size, mode="train", flag_ddp=cfg.model.mp)
    test_loader = get_dataloader(era5_test_dataset, 2, mode="test", flag_ddp=cfg.model.mp)
    logger.info(f"Dataloader done!, the number of train/test samples is \
        {len(train_loader)*cfg.hyper_params.batch_size}/{len(test_loader)*cfg.hyper_params.batch_size}")

    # loss
    loss_func = LOSSES[cfg.hyper_params.loss]
    least_loss = None
    patience = cfg.hyper_params.PATIENCE 
    
    # main
    logger.info("============== Start training ==============")
     
    for epoch in range(start_epoch, cfg.hyper_params.EPOCH + 1):
        if cfg.model.mp:
            train_loader.sampler.set_epoch(epoch)
        start0 = time.time()
        iters = len(train_loader)
        # train
        train_loss_tensor, scheduler = train_one_epoch(rank, ema, model, vae_model, diffusion, train_loader, loss_func, optimizer, scheduler, writer, cfg, epoch)        
        # test
        test_loss_tensor = test_one_epoch(rank, ema, vae_model, diffusion_sampling, test_loader, loss_func, img_base_path, logger, cfg, epoch, iters)
        
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        
        flag_break = torch.zeros(1).cuda(rank)
        if dist.get_rank() == 0: 
            train_loss =  train_loss_tensor.cpu().numpy()/dist.get_world_size()
            test_loss =  test_loss_tensor.cpu().numpy()/dist.get_world_size()
         
            writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, epoch)
            
            logger.info(f"Epoch: {epoch}, train loss: {train_loss}, test loss: {test_loss}")
            logger.info(f"Training one epoch costs: {(time.time() - start0)/60} minutes")
            # early stop
            
            if least_loss is None:
                least_loss = test_loss + 0
            if least_loss < test_loss:
                patience -= 1
                logger.info(f'Test loss stop decreasing!!!, patience is {patience}')
            else:
                least_loss = test_loss + 0
                patience = cfg.hyper_params.PATIENCE  
                save_diffusion_model(epoch, ema, model, optimizer, scheduler, train_loss, test_loss, model_path, cfg, best=True)
            if epoch % 1 == 0:
                save_diffusion_model(epoch, ema, model, optimizer, scheduler, train_loss, test_loss, model_path, cfg)
            if patience <= 0:
                flag_break += 1
                logger.info('Early stop!!!')
                break
        dist.all_reduce(flag_break, op=dist.ReduceOp.SUM)
        if flag_break == 1:
            break
    logger.info("============== End training ==============")
        
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
    
