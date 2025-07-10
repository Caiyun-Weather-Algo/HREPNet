import arrow
import pickle
import numpy as np
from einops import rearrange
from src.datasets.load_data import load_ec_data, load_static
from src.utils.feature_encoding import get_coords, get_time_features


class ECDataset():

    def __init__(self,
                 start_time,
                 end_time,
                 lead_time,
                 start_interval_hour,  
                 input_vars, 
                 output_vars, 
                 mode="train", 
                 input_step=1,
                 start_lead=0,
                 forecast_step=6, 
                 sample_interval=1, 
                 viz_vars=False,
                 stats_file=f"../data/era5_stats_1980-2018.pkl", # variables of loaded data
                 norm_method="minmax", 
                 use_static=False, 
                 add_latlon_time=False, 
                 preprocess=False, 
                 fcst_tp=False, 
                 target="tp_era5", 
                 on_cloud=False,
                 ec_dir="",
                 ):
        
        # varia le configuration and get pre-calculated mean, variance, minimum, and maximum values to corresponding variables for standardization
        with open(stats_file, 'rb') as pickle_file:
            self.data_stats = pickle.load(pickle_file)
        # self.variables = OrderedDict(load_yaml_file(variable_config_file))
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.fcst_tp = fcst_tp
        self.on_cloud = on_cloud
        self.type = type 
        
        # parameters
        self.param_sfc = ["mslp", "10m_u_component_of_wind", "10m_v_component_of_wind", "2mt"]
        self.param_pl = ["geopotential", "specific_humidity", "temperature", "u_component_of_wind", "v_component_of_wind"]        
        
        # retrieve the index of data based on the specified order of variables
        surf_idxs = [self.param_sfc.index(k) for k in input_vars["surface"]]
        high_idxs = [self.param_pl.index(k) for k in input_vars["high"]]
        self.input_var_idxs = {"surf": surf_idxs, "high": high_idxs}
        # surf_idxs = [list(self.variables["surface"].keys()).index(k) for k in output_vars["surface"]]
        # high_idxs = [list(self.variables["high"].keys()).index(k) for k in output_vars["high"]]
        # self.output_var_idxs = {"surf": surf_idxs, "high": high_idxs}
        
        era5_levels_norm = self.data_stats["levels"]
        era5_levels_gc = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

        self.input_lev_idxs_norm = [era5_levels_norm.index(j) for j in self.input_vars["levels"]]
        self.input_lev_idxs_gc = [era5_levels_gc.index(j) for j in self.input_vars["levels"]]

        # input length & output length
        self.start_time = start_time 
        self.end_time = end_time
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
        self.surf_data, self.high_data = load_ec_data(ec_dir, self.start_time, end_time, start_interval_hour)
        
        # data time
        self.sample_interval = sample_interval
        self.lead_time = lead_time
        self.init_times = list(self.surf_data.keys())  # get all initiate times
        self.total_samples = self._calculate_total_samples()
        self.num_samples_per_time = self.lead_time - self.input_step + 2

        self.t1980 = int(arrow.get("1980010100", "YYYYMMDDHH").timestamp())
                
        if use_static:
            self.z = load_static("z")
            self.lsm = load_static("lsm")
            self.slt = load_static("slt")
            self.z = (self.z-self.z.min())/(self.z.max()-self.z.min())
            self.slt = (self.slt-self.slt.min())/(self.slt.max()-self.slt.min())

        self.coords = get_coords(degree=False)

    def _calculate_total_samples(self):
        total_samples = 0
        for init_time in self.init_times:
            total_samples += (self.lead_time +1 - self.input_step + 1)  # 每个起报时次生成 (lead_time - 1) 个样本
        return total_samples    
    
    def get_idx(self, timestamp):
        yr = arrow.get(int(timestamp)).format("YYYY")
        yr_idx = int(eval(yr)-1980)
        idx = int((timestamp - (arrow.get(yr, "YYYY").timestamp()))//3600)
        return yr_idx, idx
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        init_idx = idx // self.num_samples_per_time  
        step_idx = idx % self.num_samples_per_time + self.input_step - 1 
        init_time = self.init_times[init_idx]
        init_time_stamp = arrow.get(init_time, "YYYYMMDDHH").timestamp() 
        forecast_time_stamp = init_time_stamp + 3600*(step_idx)
        x_surf = self.surf_data[init_time]['sfc'][(step_idx - self.input_step + 1): (step_idx + 1)].to_numpy()
        x_high = self.high_data[init_time]['high'][(step_idx - self.input_step + 1): (step_idx + 1)].to_numpy()

        # select data
        x_surf = x_surf[:, self.input_var_idxs["surf"]]
        x_high = x_high[:, self.input_var_idxs["high"]][:, :, self.input_lev_idxs_gc]
        
        # normalization 
        x_surf = np.concatenate([self.norm(x_surf[:, k:k+1], "surface", self.input_vars["surface"][k], level_idx=None, norm_method=self.norm_method) 
                                 for k in range(x_surf.shape[1])], axis=1)
        for j, level_idx in enumerate(self.input_lev_idxs_norm):
            x_high[:, :, j] = np.concatenate([self.norm(x_high[:, k:k+1, j], "high", self.input_vars["high"][k], level_idx, norm_method=self.norm_method) 
                                            for k in range(x_high.shape[1])], axis=1)
        x_high[np.isnan(x_high)] = 0             
        x_surf = x_surf.astype(np.float32)
        x_high = x_high.astype(np.float32)
        
        x_surf0 = x_surf[-1]
        x_high0 = rearrange(x_high[-1], 'v d h w -> (v d) h w')
        
        x_surf = rearrange(x_surf, 't v h w -> (t v) h w')
        x_high = rearrange(x_high, 't v d h w -> (t v d) h w')

        if self.use_static:
            x_surf = np.concatenate((x_surf, self.z, self.lsm, self.slt), axis=0)
            
        # lat - lon data 
        if self.add_latlon_time:
            time_features = get_time_features(arrow.get(int(forecast_time_stamp)).format('YYYY-MM-DD HH:00'))
            time_features = np.broadcast_to(time_features[:, np.newaxis, np.newaxis], 
                                            (time_features.shape[0], x_surf.shape[1], x_surf.shape[2]))
            
            x_surf = np.concatenate((x_surf, self.coords, time_features), axis=0)
        x_surf = x_surf.astype(np.float32)
        return (init_time_stamp, forecast_time_stamp, [x_surf, x_high], [x_surf0, x_high0])
        
    def norm(self, data, level, name, level_idx=None, norm_method="minmax"):
        if norm_method=="minmax":
            minmax = self.data_stats[level][name]["minmax"]
        else:
            minmax = self.data_stats[level][name]["meanstd"]
        
        if level_idx is not None:
            minmax = minmax[:, level_idx]          

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
        