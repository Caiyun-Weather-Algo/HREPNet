import arrow
import time
import pickle
from pathlib import Path
import numpy as np
import yaml
import os
from einops import rearrange
from omegaconf import DictConfig

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
import sys 
sys.path.append(project_path)

from src.utils.data_utils import tp2dbz
from src.datasets.load_data import load_era5_data, load_static
from src.utils.feature_encoding import get_coords, get_time_features


class DTMDataset():
    """
    A dataset class for loading and processing ERA5 reanalysis data for precipitation forecasting.
    
    This class handles loading and preprocessing of ERA5 reanalysis data, including surface variables,
    pressure level variables, and total precipitation data. It supports data normalization, time feature
    encoding, and static data integration (terrain height, land-sea mask, soil type).
    
    Args:
        start_time (str): Start time in format 'YYYY-MM-DD_HH'
        end_time (str): End time in format 'YYYY-MM-DD_HH'
        valid_time (str): Validation start time in format 'YYYY-MM-DD_HH'
        test_time (str): Test start time in format 'YYYY-MM-DD_HH'
        input_vars (dict): Dictionary containing input variables configuration for surface and pressure levels
        output_vars (dict): Dictionary containing output variables configuration for surface and pressure levels
        mode (str): Dataset mode, one of ['train', 'valid', 'test']
        input_step (int): Number of historical timesteps to use (default: 1)
        start_lead (int): Lead time for forecast start (default: 0)
        forecast_step (int): Number of future timesteps to forecast (default: 6)
        sample_interval (int): Interval between samples in hours (default: 1)
        viz_vars (bool): Whether to enable visualization mode (default: False)
        stats_file (str): Path to statistics file for data normalization (default: 'data/era5_stats_1980-2018.pkl')
        norm_method (str): Normalization method, one of ['minmax', 'meanstd'] (default: 'minmax')
        use_static (bool): Whether to include static data (terrain, land-sea mask, soil type) (default: False)
        add_latlon_time (bool): Whether to add latitude, longitude and time features (default: False)
        preprocess (bool): Whether to preprocess the data (default: False)
        fcst_tp (bool): Whether to forecast total precipitation (default: False)
        target (str): Target variable name (default: 'tp_era5')
    
    Returns:
        tuple: (timestamp, [x_surf, x_high], y_tp) where:
            - timestamp: Sample start time
            - x_surf: Surface variables data
            - x_high: Pressure level variables data
            - y_tp: Target precipitation data
    """

    def __init__(self,
                 start_time,
                 end_time,
                 valid_time,
                 test_time,  
                 input_vars, 
                 output_vars, 
                 mode="train", 
                 input_step=1,
                 start_lead=0,
                 forecast_step=6, 
                 sample_interval=1, 
                 viz_vars=False,
                 stats_file=f"{project_path}/data/era5_stats_1980-2018.pkl", # variables of loaded data
                 norm_method="minmax", 
                 use_static=False, 
                 add_latlon_time=False, 
                 preprocess=False, 
                 fcst_tp=False, 
                 target="tp_era5",
                 era5_path='',
                 ):
        
        # variable configuration and get pre-calculated mean, variance, minimum, and maximum values to corresponding variables for standardization
        with open(stats_file, 'rb') as pickle_file:
            self.data_stats = pickle.load(pickle_file)
            
        # self.variables = OrderedDict(load_yaml_file(variable_config_file))
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.fcst_tp = fcst_tp
        
        # retrieve the index of data based on the specified order of variables
        surf_idxs = [list(self.data_stats["surface"].keys()).index(k) for k in input_vars["surface"]]
        high_idxs = [list(self.data_stats["high"].keys()).index(k) for k in input_vars["high"]]
        self.input_var_idxs = {"surf": surf_idxs, "high": high_idxs}
        
        era5_levels_norm = self.data_stats["levels"]
        era5_levels_gc = [1000, 950, 850, 700, 600, 500, 450, 400, 300, 250, 200, 150, 100, 925, 50]
        self.input_lev_idxs_norm = [era5_levels_norm.index(j) for j in self.input_vars["levels"]]
        self.input_lev_idxs_gc = [era5_levels_gc.index(j) for j in self.input_vars["levels"]]

        # input length & output length
        self.forecast_step = forecast_step
        self.input_step = input_step
        self.start_lead = start_lead 
        
        # 
        self.viz_vars = viz_vars
        self.mode = mode
        self.norm_method = norm_method
        self.use_static = use_static
        self.add_latlon_time = add_latlon_time
        self.preprocess = preprocess
        self.target = target
        
        # load data
        self.surf_data, self.high_data, self.high_data_925, self.tp_data = load_era5_data(era5_path)

        # data time
        self.sample_interval = sample_interval
        self.sample_start_t = self.gen_sample_times(start_time, end_time, valid_time, test_time)
        self.t1980 = int(arrow.get("1980010100", "YYYYMMDDHH").timestamp())
        self.t2016 = int(arrow.get("2016010100", "YYYYMMDDHH").timestamp())
                
        if use_static:
            self.z = load_static("z")
            self.lsm = load_static("lsm")
            self.slt = load_static("slt")
            self.z = (self.z-self.z.min())/(self.z.max()-self.z.min())
            self.slt = (self.slt-self.slt.min())/(self.slt.max()-self.slt.min())

        self.coords = get_coords(degree=False)

    def gen_sample_times(self, start_time, end_time, valid_time, test_time):
        start_time_stamp = int(arrow.get(start_time, "YYYY-MM-DD_HH").timestamp())
        end_time_stamp = int(arrow.get(end_time, "YYYY-MM-DD_HH").timestamp())
        sample_start_t = list(range(start_time_stamp, end_time_stamp, 3600))
        n = len(sample_start_t)
        
        valid = int(arrow.get(valid_time, "YYYY-MM-DD_HH").timestamp())
        test = int(arrow.get(test_time, "YYYY-MM-DD_HH").timestamp())
        valid_idx = sample_start_t.index(valid)
        test_idx = sample_start_t.index(test)
        
        if self.mode == "train":
            sample_start_t = sample_start_t[:valid_idx][::self.sample_interval]
        elif self.mode == "valid":
            sample_start_t = sample_start_t[valid_idx: test_idx]
        elif self.mode == "test":
            sample_start_t = sample_start_t[test_idx:]
        else:
            raise ValueError("mode argument error")
        return sample_start_t             
    
    def get_idx_high(self, timestamp):
        yr = arrow.get(int(timestamp)).format("YYYY")
        yr_idx = int(eval(yr)-1980)
        idx = int((timestamp - (arrow.get(yr, "YYYY").timestamp()))//3600)
        return yr_idx, idx
        
    def __len__(self):
        return len(self.sample_start_t) - self.input_step + 1
    
    def __getitem__(self, idx):
        idx = idx + self.input_step - 1
        sample_start_time = self.sample_start_t[idx]
        sample_index = int((sample_start_time - self.t1980)//3600)
        tp_sample_index = int((sample_start_time - self.t1980)//3600)
        
        x_surf = self.surf_data['2m_temperature'][(sample_index - self.input_step + 1): (sample_index + 1)].to_numpy()
        x_high = []
        for high_idx in range(idx - self.input_step + 1, idx + 1):
            high_sample_start_time = self.sample_start_t[high_idx]
            yr_idx, t_idx = self.get_idx_high(high_sample_start_time)
            high1 = self.high_data[yr_idx]['geopotential'][t_idx].to_numpy()
            high2 = self.high_data_925[yr_idx]['geopotential'][t_idx].to_numpy()
            x_high.append(np.concatenate((high1, high2), axis=1)[:, self.input_lev_idxs_gc])
        x_high = np.array(x_high)

        y_tp = self.tp_data["total_precipitation"][(tp_sample_index + self.start_lead): (tp_sample_index  + self.start_lead + self.forecast_step)].to_numpy()
        y_tp = tp2dbz(y_tp*1000)
      
        # normalization 
        x_surf = np.concatenate([self.norm(x_surf[:, k:k+1], "surface", self.input_vars["surface"][k], level_idx=None, norm_method=self.norm_method) 
                                 for k in range(x_surf.shape[1])], axis=1)
        for j, level_idx in enumerate(self.input_lev_idxs_norm):
            x_high[:, :, j] = np.concatenate([self.norm(x_high[:, k:k+1, j], "high", self.input_vars["high"][k], level_idx, norm_method=self.norm_method) 
                                            for k in range(x_high.shape[1])], axis=1)
        x_high[np.isnan(x_high)] = 0             
        x_surf = x_surf.astype(np.float32)
        x_high = x_high.astype(np.float32)
        x_surf = rearrange(x_surf, 't v h w -> (t v) h w')
        x_high = rearrange(x_high, 't v d h w -> (t v d) h w')

        # x = np.concatenate((x_surf, x_high), axis=0)

        if self.use_static:
            x_surf = np.concatenate((x_surf, self.z, self.lsm, self.slt), axis=0)
            
        # lat - lon data 
        if self.add_latlon_time:
            time_features = get_time_features(arrow.get(int(sample_start_time)).format('YYYY-MM-DD HH:00'))
            time_features = np.broadcast_to(time_features[:, np.newaxis, np.newaxis], 
                                            (time_features.shape[0], x_surf.shape[1], x_surf.shape[2]))
            
            x_surf = np.concatenate((x_surf, self.coords, time_features), axis=0)
        x_surf = x_surf.astype(np.float32)

        if self.mode == "train":
            return (sample_start_time, [x_surf, x_high], y_tp)
        else:
            return (sample_start_time, [x_surf, x_high], y_tp)

    def norm(self, data, level, name, level_idx=None, norm_method="minmax"):
        if norm_method=="minmax":
            minmax = self.data_stats[level][name]["minmax"]
        else:
            minmax = self.data_stats[level][name]["meanstd"]
        
        if level_idx is not None:
            minmax = minmax[:, level_idx]          
        # print(level, name, level_idx, data.max(), data.min(), minmax)

        if norm_method=="minmax":
            data[data<minmax[1]] = minmax[1]
            data[data>minmax[0]] = minmax[0]
            normed = (data - minmax[1]) / (minmax[0] - minmax[1])
        else:
            normed = (data-minmax[0])/minmax[1]
        return normed
    
    def denorm(self, data, level, name, level_idx=None, norm_method="minmax"):
        if norm_method=="minmax":
            minmax = self.data_stats[level][name]["minmax"]
        else:
            minmax = self.data_stats[level][name]["meanstd"]
        
        if level_idx is not None:
            minmax = minmax[:, level_idx]
        
        if norm_method=="minmax":
            denormed = data * (minmax[0] - minmax[1]) + minmax[1]
        else:
            denormed = data * minmax[1] + minmax[0]
        return denormed
        
 
def main(): 
    cfg = yaml.load(open(f"{project_path}/config/train_tp_config.yaml"), Loader=yaml.Loader)
    cfg = DictConfig(cfg)

    mode = "test"
    era5_loader = DTMDataset(
                            start_time=cfg.start_time,
                            end_time="2021-08-01_00",
                            valid_time="2020-07-01_00",
                            test_time="2021-07-01_00",
                            input_vars=cfg.input,
                            output_vars=cfg.output,
                            mode=mode,
                            input_step=cfg.hyper_params.input_step,
                            forecast_step=cfg.hyper_params.forecast_step,
                            start_lead=0,  # cfg.hyper_params.start_lead,
                            sample_interval=cfg.hyper_params.sample_interval,
                            norm_method=cfg.hyper_params.norm_method, 
                            preprocess=True, 
                            target="tp_era5",
                            )

    img_path = Path("sample_images")
    img_path.mkdir(exist_ok=True, parents=True)
    n = len(era5_loader)
    print("sample num:", n)
    from src.utils.plot_util import plot_single_var, plot_cloud
    pressures = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    name = ['z', 't', 'sp', 'u', 'v']
    name_sfc = ['2mt', 'mslp', '10u', '10v']

    for k in range(0, n, 24*10):
        t0 = time.time()
        time_stamp, x, ytp = era5_loader[k]
        ymdh = arrow.get(time_stamp).format('YYYY-MM-DD_HH')
        print(ymdh, x[1].shape, ytp.shape, ytp.max(), ytp.min())
        print("load one sample costs:", time.time()-t0)
        
        for j in range(4):
            plot_single_var(f"all_{ymdh}_{name_sfc[j]}", x[0][j, ::-1], region='era50')
            print(f"all_{ymdh}_{name_sfc[j]}, {x[0][j, ::-1].mean(), x[0][j, ::-1].std()}")

        for j in range(5):
            for ip in range(13):
                plot_single_var(f"all_{ymdh}_{name[j]}_{pressures[ip]}", x[1][ip + j*13, ::-1], region='era50')
                print(f"all_{ymdh}_{name[j]}_{pressures[ip]}, {x[1][ip + j*13, ::-1].mean(), x[1][ip + j*13, ::-1].std()}")
        print("load one sample costs:", time.time()-t0)
        

if __name__=="__main__":
    #import torch_xla.distributed.xla_multiprocessing as xmp
    #xmp.spawn(main, args=(), nprocs=None)
    main()