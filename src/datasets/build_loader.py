import torch
from torch.utils.data import DataLoader
from src.datasets.dtm_dataset import DTMDataset     
from src.datasets.dtm_hr_dataset import DTM_HR_Dataset
from src.datasets.prob_dataset import ProbDataset
from src.datasets.ec_dataset import ECDataset


def get_dataset(mode, cfg):
    if cfg.dataset == 'prob':
        dataset = ProbDataset(
                start_time=cfg.start_time,
                end_time=cfg.end_time,
                valid_time=cfg.valid_time,
                test_time=cfg.test_time,
                input_vars=cfg.input,
                output_vars=cfg.output,
                mode=mode,
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
    elif cfg.dataset == 'dtm':
         dataset = DTMDataset(
                start_time=cfg.start_time,
                end_time=cfg.end_time,
                valid_time=cfg.valid_time,
                test_time=cfg.test_time,
                input_vars=cfg.input,
                output_vars=cfg.output,
                mode=mode,
                input_step=cfg.hyper_params.input_step,
                forecast_step=cfg.hyper_params.forecast_step, 
                start_lead=cfg.hyper_params.start_lead, 
                sample_interval=cfg.hyper_params.sample_interval,
                use_static=cfg.hyper_params.use_static, 
                add_latlon_time=cfg.hyper_params.add_latlon_time,
                norm_method=cfg.hyper_params.norm_method,
                era5_path=cfg.era5_path,
         )
    elif cfg.dataset == 'dtm_hr':
        dataset = DTM_HR_Dataset(
                start_time=cfg.start_time,
                end_time=cfg.end_time,
                valid_time=cfg.valid_time,
                test_time=cfg.test_time,
                input_vars=cfg.input,
                output_vars=cfg.output,
                mode=mode,
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
    if cfg.dataset == 'ec':
        dataset = ECDataset(
                start_time=cfg.start_time,
                end_time=cfg.end_time,
                lead_time=cfg.lead_time,
                start_interval_hour=cfg.start_interval_hour,
                input_vars=cfg.input,
                output_vars=cfg.output,
                mode=mode,
                input_step=cfg.hyper_params.input_step,
                forecast_step=cfg.hyper_params.forecast_step, 
                start_lead=cfg.hyper_params.start_lead, 
                sample_interval=cfg.hyper_params.sample_interval,
                use_static=cfg.hyper_params.use_static, 
                add_latlon_time=cfg.hyper_params.add_latlon_time,
                norm_method=cfg.hyper_params.norm_method,
                stats_file=cfg.stats_file,
                on_cloud=cfg.on_cloud,
                ec_dir=cfg.ec_path,
                )
    return dataset


def get_dataloader(dataset, batch_size, mode="train", flag_ddp=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        
    if flag_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=4, 
                            sampler=sampler, 
                            # multiprocessing_context="spawn", 
                            )    
    return dataloader