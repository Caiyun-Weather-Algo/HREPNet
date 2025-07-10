import os
import time
import arrow
import yaml
import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils.model_utils import get_optimizer, get_scheduler, load_checkpoint, save_model, LOSSES
from src.models.build_model import build_model
from src.utils.plot_util import plot_train_result
from src.utils.dist import setup, cleanup
from src.utils.logger import create_logger
from src.datasets.build_loader import get_dataset, get_dataloader

     
def train_one_epoch(rank, model, train_loader, loss_func, optimizer, scheduler, writer, cfg, epoch):
    iters = len(train_loader)
    train_loss = torch.tensor(0.0).to(rank)
    model.train()

    for i, data in enumerate(train_loader):   
        timestamp, input, target = data
        input = [input[0].to(rank), input[1].to(rank)]
        target = target.to(rank)  
        # forward
        output = model(input)
        # calculate loss
        loss = loss_func(output, target)    
        train_loss += loss.detach()
        # backward
        loss.backward()
        # update paramters after `grad_accum` iterations
        if i%cfg.hyper_params.grad_accum==0:
            optimizer.step()
            optimizer.zero_grad()
            if cfg.hyper_params.scheduler=="cosineannealingwarmrestart" or cfg.hyper_params.scheduler=="cosineannealing":
                scheduler.step(epoch + i / iters)
    train_loss = train_loss / len(train_loader)
    return train_loss, scheduler    
 
           
def test_one_epoch(rank, model, test_loader, loss_func, img_base_path, logger, cfg, epoch):
    model.eval()
    test_loss = torch.tensor(0.0).to(rank)
    for i, data in enumerate(test_loader):
        timestamp, input, target = data

        input = [input[0].to(rank), input[1].to(rank)]
        target = target.to(rank)  
        
        with torch.no_grad():               
            output = model(input)
            loss = loss_func(output, target)
            
        test_loss += loss.detach()
        # plot mid-results
        if i % cfg.hyper_params.verbose_step == 0 and (epoch-1) % 2==0:
            t0 = time.time()
            img_path = img_base_path+f'/epoch{epoch}'
            os.makedirs(img_path, exist_ok=True)
            if (cfg.model.mp and dist.get_rank() == 0) or not cfg.model.mp:
                timestamp = timestamp.detach().cpu().numpy()
                ymdh = arrow.get(int(timestamp[0])).format('YYYY-MM-DD_HH')
                plot_train_result(f"{img_path}/tp_era5_{ymdh}", 
                                  "radar",
                                  output.detach().cpu().numpy()[0, 0, ::-1], 
                                  target.detach().cpu().numpy()[0, 0, ::-1],
                                  region='era50',
                                  )
                logger.info(f"saving images costs: {time.time() - t0}")
    test_loss = test_loss / len(test_loader) 
    return test_loss


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
    
    # model DDP
    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)         
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)  
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])   
    
    # Dataloader
    era5_train_dataset = get_dataset(mode="train", cfg=cfg)
    era5_test_dataset = get_dataset(mode="valid", cfg=cfg)
    train_loader = get_dataloader(era5_train_dataset, cfg.hyper_params.batch_size, mode="train", flag_ddp=cfg.model.mp)
    test_loader = get_dataloader(era5_test_dataset, cfg.hyper_params.batch_size, mode="valid", flag_ddp=cfg.model.mp)
    logger.info(f"Dataloader done!, the number of train/test samples is \
        {len(train_loader)*cfg.hyper_params.batch_size}/{len(test_loader)*cfg.hyper_params.batch_size}")

    # loss
    loss_func = LOSSES[cfg.hyper_params.loss]
    least_loss = None
    patience = cfg.hyper_params.PATIENCE 
    # main
    logger.info("============== Start training ==============")
    
    # encoding
    for epoch in range(start_epoch, cfg.hyper_params.EPOCH + 1):
        if cfg.model.mp:
            train_loader.sampler.set_epoch(epoch)
        start0 = time.time()
        # train
        train_loss_tensor, scheduler = train_one_epoch(rank, model, train_loader, loss_func, optimizer, scheduler, writer, cfg, epoch)
        # test
        test_loss_tensor = test_one_epoch(rank, model, test_loader, loss_func, img_base_path, logger, cfg, epoch)

        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        
        flag_break = torch.zeros(1).cuda(rank)
        if dist.get_rank() == 0: 
            train_loss = train_loss_tensor.cpu().numpy()/dist.get_world_size()
            test_loss = test_loss_tensor.cpu().numpy()/dist.get_world_size()
         
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
                save_model(epoch, model, optimizer, scheduler, train_loss, test_loss, model_path, cfg, best=True)
            if epoch % 1 == 0:
                save_model(epoch, model, optimizer, scheduler, train_loss, test_loss, model_path, cfg)
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