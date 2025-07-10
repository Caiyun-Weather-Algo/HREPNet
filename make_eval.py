import os
import time
import arrow
import yaml
import hydra
from omegaconf import OmegaConf
from typing import OrderedDict
import numpy as np
import pandas as pd 
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models.build_model import build_model
# from evaluation.evaluation import resume, mid_rmse
from src.utils.result_util import data_to_zarr, data_to_npz
from src.utils.plot_util import plot_train_result
from src.datasets.build_loader import get_dataloader, get_dataset
from src.utils.dist import setup, cleanup
from src.utils.feature_encoding import get_posi_encoding

variables = OrderedDict(yaml.load(open("./config/era5_variables.yaml", "r"), Loader=yaml.Loader))


def test_loop(rank,  model, test_loader, cfg, thresholds):
    model.eval()
    start = time.time()
    counts = torch.zeros((len(thresholds), 4), dtype=torch.float32).cuda(rank)

    eval_path = f"{cfg.base_path}/{cfg.model.name}_exp{cfg.exp}/eval_paper"
    os.makedirs(eval_path, exist_ok=True)
    
    for i, data in enumerate(test_loader):
        timestamp, input, target = data
        # timestamp = timestamp.cpu().numpy()
        # ymdh = arrow.get(int(timestamp[0])).format('YYYY-MM-DD_HH')
        # input = torch.cat((input[0], input[1]), dim=1)
        input = [input[0].to(rank), input[1].to(rank)]  

        target = target.to(rank) 
        # x_tp = x_tp.to(rank)
 
        with torch.no_grad():               
            output = model(input)
            
        target = target.cpu().numpy() * 55
        output = output.cpu().numpy()   
        # target = dbz2tp(target)
        # output = dbz2tp(output)
        
        # store data
        for data, t in zip(output, timestamp.cpu().numpy()):
            ymdh = arrow.get(int(t)).format('YYYY-MM-DD_HH')
            # data_to_zarr(data, t, f'{cfg.base_path}/pred/era5')
            data_to_npz(data, ymdh, ymdh, f'{cfg.base_path}/data/pred/era5-dtm')
        
        # evaluation
        if (i % cfg.hyper_params.verbose_step == 0) & (dist.get_rank() == 0):
            ymdh = arrow.get(int(timestamp[0])).format('YYYY-MM-DD_HH')
            plot_train_result(f"{eval_path}/tp_era5_{ymdh}", "rainfall", output[0, 0, ::-1], target[0, 0, ::-1], region='era50')
        
        for j, v in enumerate(thresholds):
            counts[j,0] += torch.tensor(((target >= v) & (output >= v)).sum(), dtype=torch.float32)  # hit
            counts[j,1] += torch.tensor(((target >= v) & (output < v)).sum(), dtype=torch.float32) # missing
            counts[j,2] += torch.tensor(((target < v) & (output >= v)).sum(), dtype=torch.float32) # false
            counts[j,3] += torch.tensor(((target < v) & (output < v)).sum(), dtype=torch.float32)
    # sum counts on different gpus
    dist.reduce(counts, dst=0, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        csi = counts[:, 0]/(counts[:, :3].sum(axis=1))
        pod = counts[:, 0]/(counts[:, 0] + counts[:, 1])
        far = counts[:, 2]/(counts[:, 0] + counts[:, 2])
        scores = np.array([csi.cpu().numpy(), pod.cpu().numpy(), far.cpu().numpy()], dtype=np.float32)
        counts = counts.cpu().numpy()
        df_res = pd.DataFrame(scores, index=['CSI', 'POD', 'FAR'], columns=thresholds)
        # df_res.astype(float).round(3).to_csv(f"{eval_path}/exp{cfg.exp}_score.csv")
        df_res.to_csv(f"{eval_path}/exp{cfg.exp}_score.csv", float_format="%.3f")
        print(df_res)


def run_eval(rank, world_size, cfg):
    thresholds = [0.1, 2, 5, 10, 15, 20]
    model = build_model(cfg.model.name, cfg)
    if cfg.model.mp:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        print("gpu rank:", torch.distributed.get_rank(), "gpu total:", torch.cuda.device_count())
    
    device = torch.device("cuda", rank)
    model = model.to(device)
    ckpt = torch.load(f"{cfg.base_path}/{cfg.model.name}_exp{cfg.exp}/models/{cfg.model.name}_epoch30.pth.tar", map_location="cuda:0")
   
    model.load_state_dict(ckpt["model_state_dict"])

    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # dataset
    era5_test_dataset = get_dataset(mode="test", cfg=cfg)
    # data loader
    test_loader = get_dataloader(era5_test_dataset, cfg.hyper_params.batch_size, mode="test", flag_ddp=cfg.model.mp)

    # test
    test_loop(rank, model, test_loader, cfg, thresholds)
    
    cleanup()


@hydra.main(version_base=None, config_path="./config", config_name="train_tp_config.yaml")
def main(cfg):
    if cfg.model.mp:
        world_size = 2
        mp.spawn(run_eval,
                args=(world_size, cfg),
                nprocs=2,
                join=True)
    else:
        run_eval(rank=0, world_size=1, cfg=cfg)


if __name__ == '__main__':
    main()
