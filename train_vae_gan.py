import os
import time
import arrow
import yaml
import hydra
from omegaconf import OmegaConf
from typing import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils.model_utils import  get_scheduler, load_checkpoint, save_gan_model, LOSSES
from src.models.build_model import build_model
from src.utils.plot_util import plot_train_result
from src.utils.dist import setup, cleanup
from src.utils.logger import create_logger
from src.datasets.build_loader import get_dataset, get_dataloader

variables = OrderedDict(yaml.load(open("./config/era5_variables.yaml", "r"), Loader=yaml.Loader))

     
def train_one_epoch(rank, model, train_loader, loss_func, optimizers, schedulers, writer, cfg, epoch):
    iters = len(train_loader)
    optimizer_ae, optimizer_disc = optimizers
    scheduler_ae, scheduler_disc = schedulers
    train_loss = torch.tensor(0.0).to(rank)
    model.train()

    for i, data in enumerate(train_loader): 
        print(f"epoch {epoch}, iter {i}")
        # _, input = data
        x_surf, x_high, era5_tp, cmpa_tp = data

        input = pre_processing(cmpa_tp, rank, cfg.model.name)
        # train autoencoder
        optimizer_ae.zero_grad()
        if torch.cuda.device_count() > 1:
            aeloss, log_dict_ae = model.module.training_step(input, optimizer_idx=0, global_step=epoch*iters + i)
        else:
            aeloss, log_dict_ae = model.training_step(input, optimizer_idx=0, global_step=epoch*iters + i)
        aeloss.backward()
        optimizer_ae.step()

        # train discriminator
        optimizer_disc.zero_grad()
        if torch.cuda.device_count() > 1:
            discloss, log_dict_disc = model.module.training_step(input, optimizer_idx=1, global_step=epoch*iters + i)  
        else:
            discloss, log_dict_disc = model.training_step(input, optimizer_idx=1, global_step=epoch*iters + i)  
        discloss.backward()
        optimizer_disc.step()

        if cfg.hyper_params.scheduler=="cosineannealingwarmrestart" or cfg.hyper_params.scheduler=="cosineannealing":
            scheduler_ae.step(epoch + i / iters)
            scheduler_disc.step(epoch + i / iters)
            
        train_loss += aeloss.detach()
    train_loss = train_loss / len(train_loader)
    return train_loss, [scheduler_ae, scheduler_disc]    
 
           
def test_one_epoch(rank, model, test_loader, loss_func, img_base_path, logger, cfg, epoch, iters): 
    model.eval()
    test_loss = torch.tensor(0.0).to(rank)
    for i, data in enumerate(test_loader):
        timestamp, x_surf, x_high, era5_tp, cmpa_tp = data
        input = pre_processing(cmpa_tp, rank, cfg.model.name)
       
        with torch.no_grad(): 
            output, posterior = model(input)
            loss, log_dict_ae = model.module.loss(input, output, posterior, 0, epoch*iters + i,
                                        last_layer=model.module.get_last_layer(), split="val")   
            
        test_loss += loss.detach()
        # plot mid-results
        if i % cfg.hyper_params.verbose_step == 0 and (epoch-1) % 2==0:
            t0 = time.time()
            img_path = img_base_path+f'/epoch{epoch}'
            os.makedirs(img_path, exist_ok=True)
            if (cfg.model.mp and dist.get_rank()==0) or not cfg.model.mp:
                timestamp = timestamp.detach().cpu().numpy()
                ymdh = arrow.get(int(timestamp[0])).format('YYYY-MM-DD_HH')
                plot_train_result(f"{img_path}/tp_era5_{ymdh}", 
                                  "radar",
                                  output.detach().cpu().numpy()[0, 0, ::-1] * 46.29922575990931, 
                                  input.detach().cpu().numpy()[0, 0, ::-1] * 46.29922575990931,
                                  region='cmpa_5km')
                logger.info(f"saving images costs: {time.time() - t0}")
    test_loss = test_loss / len(test_loader) 
    return test_loss


def pre_processing(input, rank, model_name):
    if "unet" in model_name or 'swinir' in model_name:
        input = torch.cat((input[0], input[1]), dim=1)
        input = input.to(rank)
    elif "transformer" in model_name:
        input = [input[0].to(rank), input[1].to(rank)]  
    elif 'autoencoder' in model_name:
        cmpa_tp = input.to(rank)
        input = cmpa_tp  #- torch.nn.functional.interpolate(era5_tp[..., 0:180, 0:280], scale_factor=5, mode='nearest')
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
    else:
        logger = create_logger(output_dir=result_path, dist_rank=0, name=cfg.model.name)  

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

    # optimizer and scheduler
    optimizer_ae, optimizer_disc = model.configure_optimizers(cfg.hyper_params.lr)
    scheduler_ae, scheduler_disc = get_scheduler(optimizer_ae, cfg), get_scheduler(optimizer_disc, cfg)  

    if ckpt is not None:
        optimizer_ae.load_state_dict(ckpt["optimizer_ae_state_dict"])
        optimizer_disc.load_state_dict(ckpt["optimizer_disc_state_dict"])
        scheduler_ae.load_state_dict(ckpt["scheduler_ae_state_dict"])   
        scheduler_disc.load_state_dict(ckpt["scheduler_disc_state_dict"])
   
    # model DDP
    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)                   
    
    # Dataloader
    era5_train_dataset = get_dataset(mode="train", cfg=cfg)
    era5_test_dataset = get_dataset(mode="test", cfg=cfg)
    train_loader = get_dataloader(era5_train_dataset, cfg.hyper_params.batch_size, mode="train", flag_ddp=cfg.model.mp)
    test_loader = get_dataloader(era5_test_dataset, cfg.hyper_params.batch_size, mode="test", flag_ddp=cfg.model.mp)
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
        train_loss_tensor, [scheduler_ae, scheduler_disc] = train_one_epoch(rank, model, train_loader, loss_func, [optimizer_ae, optimizer_disc], [scheduler_ae, scheduler_disc], writer, cfg, epoch)        
        # test
        test_loss_tensor = test_one_epoch(rank, model, test_loader, loss_func, img_base_path, logger, cfg, epoch, iters)
        
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
                save_gan_model(epoch, model, [optimizer_ae, optimizer_disc], [scheduler_ae, scheduler_disc], train_loss, test_loss, model_path, cfg, best=True)
            if epoch % 1 == 0:
                save_gan_model(epoch, model, [optimizer_ae, optimizer_disc], [scheduler_ae, scheduler_disc], train_loss, test_loss, model_path, cfg)
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
    if cfg.on_cloud:
        cfg.base_path = cfg.remote_path

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
    
