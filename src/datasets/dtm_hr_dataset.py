import arrow
import time
from pathlib import Path
import numpy as np
import yaml
import os
from einops import rearrange
from omegaconf import DictConfig
import pickle
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
import sys 
sys.path.append(project_path)

from src.utils.data_utils import tp2dbz
from src.datasets.load_data import load_era5_data, load_static, load_cmpa_tp, load_era5_pred_tp
from src.utils.feature_encoding import get_coords, get_time_features


class DTM_HR_Dataset():
    """
    A dataset class for high-resolution precipitation forecast using ERA5 reanalysis data and CMPA precipitation data.
    
    This class handles loading and preprocessing of ERA5 reanalysis data and CMPA (China Merged Precipitation Analysis)
    data for high-resolution precipitation forecast tasks. 
    
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
        tuple: For training mode:
            - sample_start_time: Timestamp of the sample
            - [x_surf, x_high]: List containing:
                - x_surf: Surface variables data
                - x_high: Pressure level variables data
            - cmpa_tp: CMPA precipitation data
        For validation/test mode:
            Same structure as training mode
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
                 cmpa_path='',
                 ):
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
        
        # load pred tp data
        # self.pred_tp = load_era5_pred_tp("202001", "202108")
        
        # load cmpa data
        if self.mode == "train":
            self.cmpa_tp, self.cmpa_time_idx, self.cmpa_daily_idx = load_cmpa_tp(cmpa_path, "201801", "201912")
        else:
            self.cmpa_tp, self.cmpa_time_idx, self.cmpa_daily_idx = load_cmpa_tp(cmpa_path, "202108", "202108")
  
        # data time
        self.t1980 = int(arrow.get("1980010100", "YYYYMMDDHH").timestamp())
        self.t2016 = int(arrow.get("2016010100", "YYYYMMDDHH").timestamp())
        self.t2021 = int(arrow.get("2021010100", "YYYYMMDDHH").timestamp())     
        self.sample_interval = sample_interval
        self.sample_start_t = self.gen_sample_times(start_time, end_time, valid_time, test_time)
        
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
        # sample_start_t = [arrow.get(str(t)[:10], "YYYY-MM-DD").timestamp() for t in np.concatenate(self.cmpa_time_idx)]
        # sample_start_t = [t for t in range(start_time_stamp, end_time_stamp, 3600) if self.is_cmpa_data_available(t)]
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
        sample_start_t = [t for t in sample_start_t if self.is_cmpa_data_available(t)]
        return sample_start_t             
    
    def get_idx_high(self, timestamp):
        yr = arrow.get(int(timestamp)).format("YYYY")
        yr_idx = int(eval(yr)-1980)
        idx = int((timestamp - (arrow.get(yr, "YYYY").timestamp()))//3600)
        return yr_idx, idx
    
    def get_idx_pred_tp(self, timestamp):
        yr = arrow.get(int(timestamp)).format("YYYY")
        mo = arrow.get(int(timestamp)).format("MM")
        daily_idx = int(eval(yr) - 2020) * 12 + int(mo) - 1 
        hour_idx = int((timestamp - (arrow.get(yr + mo, "YYYYMM").timestamp()))//3600) 
        return daily_idx, hour_idx
            
    def get_idx_cmpa(self, timestamp):
         # CMPA data beyond 2021 is stored in Beijing time
        if timestamp >= self.t2021:
            offset = 8
        else:
            offset = 0
        ymdh = arrow.get(int(timestamp + 3600*offset)).format("YYYYMMDDHH") # 
       
        try:
            daily_idx = (self.cmpa_daily_idx).index(ymdh[0:8])
            hour_idx = list(self.cmpa_time_idx[daily_idx]).index(np.datetime64(f'{ymdh[0:4]}-{ymdh[4:6]}-{ymdh[6:8]}T{ymdh[8:10]}'))
            return daily_idx, hour_idx
        except IndexError:
            raise ValueError(f"CMPA data missing for timestamp {ymdh}")
    
    def is_cmpa_data_available(self, timestamp):
        if timestamp >= self.t2021:
            offset = 8
        else:
            offset = 0
        ymdh = arrow.get(int(timestamp + 3600*offset)).format("YYYYMMDDHH") # 
        ymdh_dt64 = np.datetime64(f'{ymdh[0:4]}-{ymdh[4:6]}-{ymdh[6:8]}T{ymdh[8:10]}')
        
        if ymdh_dt64 in np.concatenate(self.cmpa_time_idx):
            return True
        else:
            return False
    
    def __len__(self):
        return len(self.sample_start_t) - self.input_step + 1
    
    def __getitem__(self, idx):
        t0 = time.time()
        idx = idx + self.input_step - 1
        sample_start_time = self.sample_start_t[idx]
        sample_index = int((sample_start_time - self.t1980)//3600)
        
        # cmpa
        cmpa_tp = []
        for cmpa_idx in range((idx + self.start_lead), (idx  + self.start_lead + self.forecast_step)): 
            cmpa_sample_time = self.sample_start_t[cmpa_idx]
            cmpa_daily_idx, cmpa_hour_idx = self.get_idx_cmpa(cmpa_sample_time)
            cmpa_tp.append(self.cmpa_tp[cmpa_daily_idx]['precipitation'][cmpa_hour_idx].to_numpy())  

        cmpa_tp = np.array(cmpa_tp)[:, ::-1, :]

        x_surf = self.surf_data['2m_temperature'][(sample_index - self.input_step + 1): (sample_index + 1)].to_numpy()
        x_high = []
        for high_idx in range(idx - self.input_step + 1, idx + 1):
            high_sample_start_time = self.sample_start_t[high_idx]
            yr_idx, t_idx = self.get_idx_high(high_sample_start_time)
            high1 = self.high_data[yr_idx]['geopotential'][t_idx].to_numpy()
            high2 = self.high_data_925[yr_idx]['geopotential'][t_idx].to_numpy()
            x_high.append(np.concatenate((high1, high2), axis=1)[:, self.input_lev_idxs_gc])
        x_high = np.array(x_high)
           
        # y_tp = tp2dbz(y_tp*1000)
        cmpa_tp = tp2dbz(cmpa_tp)
        cmpa_tp[np.isnan(cmpa_tp)] = 0.0

        # normalization 
        x_surf = np.concatenate([self.norm(x_surf[:, k:k+1], "surface", self.input_vars["surface"][k], level_idx=None, norm_method=self.norm_method) 
                                for k in range(x_surf.shape[1])], axis=1)
        for j, level_idx in enumerate(self.input_lev_idxs_norm):
            x_high[:, :, j] = np.concatenate([self.norm(x_high[:, k:k+1, j], "high", self.input_vars["high"][k], level_idx, norm_method=self.norm_method) 
                                            for k in range(x_high.shape[1])], axis=1)
        # x_tp = self.norm(x_tp, "surface", "total_precipitation", level_idx=None, norm_method=self.norm_method)

        x_high[np.isnan(x_high)] = 0             
        x_surf = x_surf.astype(np.float32)
        x_high = x_high.astype(np.float32)
        # x_tp = x_tp.astype(np.float32)
        cmpa_tp = cmpa_tp.astype(np.float32)

        x_surf = rearrange(x_surf, 't v h w -> (t v) h w')
        x_high = rearrange(x_high, 't v d h w -> (t v d) h w')

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
            return (sample_start_time, [x_surf, x_high], cmpa_tp)
        else:
            return (sample_start_time, [x_surf, x_high], cmpa_tp)
       
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
    era5_loader = DTM_HR_Dataset(
                            start_time="2018-01-01_00",
                            end_time="2021-08-31_23",
                            valid_time="2020-01-01_00",
                            test_time="2021-08-01_00",
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
    from utils.plot_util import plot, plot_tp_result
    pressures = [1000, 950, 850, 700, 600, 500, 450, 400, 300, 250, 200, 150, 100]
    for k in range(0, n, 24*5):
        t0 = time.time()
        time_stamp, x, era5_tp, cmpa_tp = era5_loader[k]
        ymdh = arrow.get(time_stamp).format('YYYY-MM-DD_HH')
        print(x[1].min(), x[1].max(), cmpa_tp.min(), cmpa_tp.max())
        print("load one sample costs:", ymdh, time.time()-t0)
   

if __name__=="__main__":
    #import torch_xla.distributed.xla_multiprocessing as xmp
    #xmp.spawn(main, args=(), nprocs=None)
    main()