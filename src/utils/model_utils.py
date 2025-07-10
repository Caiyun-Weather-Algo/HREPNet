import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np
import os 
from src.utils import BLoss
from utils.scheduler_warmup import WarmupLR
from src.utils import data_utils


LOSSES = {
        'mse': nn.MSELoss(reduction='mean'),
        'bmse': BLoss.BMSELoss(),
        'ssim': BLoss.SSIM(window_size=11, size_average=True),
        'mse_ssim': BLoss.SSIMLoss(),
        'bmae_radar': BLoss.BMAELoss(weights=[0.1, 0.2, 0.4, 0.5, 0.7, 1], thresholds=[0.02, 0.04, 0.1, 0.2, 0.36, 0.55]),
        'bmse_prcp': BLoss.BMSELoss(weights=[1, 1.1, 2.5, 10, 100], thresholds=[3, 5, 7, 9, 11]),
        'bmse_radar': BLoss.BMSELoss(weights=[1, 6.5, 20, 60, 300], thresholds=[2, 5, 10, 20, 30]),
        'mix_loss': BLoss.MixedLoss(),
        #'lat_weighted_mse': BLoss.LatWeightedMSE(latweights(), device=0)
        }


def get_optimizer(model, cfg):
    if isinstance(cfg.hyper_params.lr, str):
        cfg.hyper_params.lr = eval(cfg.hyper_params.lr)
    if cfg.hyper_params.optimizer=="AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=cfg.hyper_params.lr, 
                                    weight_decay=cfg.hyper_params.weight_decay, 
                                    betas=(0.9, 0.95)
                                    )
    elif cfg.hyper_params.optimizer=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=cfg.hyper_params.lr, 
                                    weight_decay=cfg.hyper_params.weight_decay, 
                                    betas=(0.9, 0.95)
                                    )
    elif cfg.hyper_params.optimizer=="rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), 
                                        lr=0.01, 
                                        alpha=0.99, 
                                        eps=1e-08, 
                                        weight_decay=0
                                        )
    else:
        print("optim not supported")
        return
    return optimizer
    

def get_scheduler(optimizer, cfg):
    if cfg.hyper_params.scheduler=="reducelronplateau":
        scheduler = ReduceLROnPlateau(optimizer, threshold=1e-8, patience=4, factor=0.5, verbose=True)
    elif cfg.hyper_params.scheduler=="cosineannealing":
        scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0, last_epoch=-1, verbose=True)
    elif cfg.hyper_params.scheduler=="cosineannealingwarmrestart":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.hyper_params.EPOCH + 1, T_mult=2)
    
    if cfg.hyper_params.warmup_lr:
        scheduler = WarmupLR(scheduler, init_lr=1e-9, num_warmup=1000, warmup_strategy="linear")
    return scheduler
    

def latweights():
    lats = np.arange(60, 0-0.1, -0.25)
    idxs = data_utils.get_interp_idxs()
    lat = lats[idxs[0]]
    lat = np.tile(lat, (256, 1)).T
    lat_weights = data_utils.lat_weight(lat)
    return lat_weights


def save_model(epoch, model, optimizer, scheduler, train_loss, valid_loss, model_path, cfg, best=False):
    if cfg.model.mp or cfg.model.tpu:
        model_state = model.module.state_dict()
    else:
        model_state = model.cpu().state_dict()
    if best:
        name = "{}/{}_best.pth.tar".format(model_path, cfg.model.name)
    else:
        name = "{}/{}_epoch{}.pth.tar".format(model_path, cfg.model.name, epoch)
    
    ckpt = {
            "epoch": epoch, 
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(), 
            "scheduler_state_dict": scheduler.state_dict(), 
            "train_loss": train_loss,
            "val_loss": valid_loss
            }
    torch.save(ckpt, name)
    return


def save_diffusion_model(epoch, ema, model, optimizer, scheduler, train_loss, valid_loss, model_path, cfg, best=False):
    if cfg.model.mp or cfg.model.tpu:
        model_state = model.module.state_dict()
    else:
        model_state = model.cpu().state_dict()
    if best:
        name = "{}/{}_best.pth.tar".format(model_path, cfg.model.name)
    else:
        name = "{}/{}_epoch{}.pth.tar".format(model_path, cfg.model.name, epoch)
    
    ckpt = {
            "epoch": epoch, 
            "model_state_dict": model_state,
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), 
            "scheduler_state_dict": scheduler.state_dict(), 
            "train_loss": train_loss,
            "val_loss": valid_loss
            }
    torch.save(ckpt, name)
    return


def save_gan_model(epoch, model, optimizers, schedulers, train_loss, valid_loss, model_path, cfg, best=False):
    if cfg.model.mp or cfg.model.tpu:
        model_state = model.module.state_dict()
    else:
        model_state = model.cpu().state_dict()
    if best:
        name = "{}/{}_best.pth.tar".format(model_path, cfg.model.name)
    else:
        name = "{}/{}_epoch{}.pth.tar".format(model_path, cfg.model.name, epoch)
    
    ckpt = {
            "epoch": epoch, 
            "model_state_dict": model_state,
            "optimizer_ae_state_dict": optimizers[0].state_dict(), 
            "optimizer_disc_state_dict": optimizers[1].state_dict(), 
            "scheduler_ae_state_dict": schedulers[0].state_dict(), 
            "scheduler_disc_state_dict": schedulers[1].state_dict(), 
            "train_loss": train_loss,
            "val_loss": valid_loss
            }
    torch.save(ckpt, name)
    return


def load_checkpoint(cfg, model_path):
    fs = os.listdir(model_path)
    if len(fs)<1:
        return [None, 1]
    epochs = sorted([eval(f.split("ch")[-1].split(".")[0]) for f in fs if "epoch" in f])
    start_epoch = epochs[-1]
    ckpt_path =  "{}/{}_epoch{}.pth.tar".format(model_path, cfg.model.name, start_epoch)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    start_epoch = checkpoint["epoch"] + 1
    return [checkpoint, start_epoch]