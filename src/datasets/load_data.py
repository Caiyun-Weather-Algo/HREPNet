import xarray as xr
import numpy as np 
import os 
from datetime import datetime, timedelta
import pandas as pd  
   
      
def load_era5_data(era5_path):
    tp = xr.open_dataset(f"{era5_path}/era5_tp_1980-2022.zarr", engine="zarr",)
    surf = xr.open_dataset(f"{era5_path}/surface_1980-2022.zarr", engine="zarr",)
    high = [xr.open_dataset(f"{era5_path}/high_{yr}.zarr", engine="zarr",) for yr in range(1980, 2022)]
    high_add = [xr.open_dataset(f"{era5_path}/high_{yr}_925-50hPa.zarr",
                            engine="zarr",) for yr in range(1980, 2022)]
    return surf, high, high_add, tp


def load_era5_pred_tp(pred_tp_path, start_date_str="202005", end_date_str="202005"):
    start_date = datetime.strptime(start_date_str, "%Y%m")
    end_date = datetime.strptime(end_date_str, "%Y%m")
    monthly_dates = pd.date_range(start_date, end_date, freq='MS')
    pred_tp = []

    for date in monthly_dates:
        # num_days = pd.Timestamp(year, month, 1).days_in_month
        yyyymm = date.strftime("%Y-%m")
        file_name = f'{pred_tp_path}/prcp_data_{yyyymm}.zarr'
        print(file_name)
        ds = xr.open_dataset(file_name, engine="zarr")
        pred_tp.append(ds)       
    return pred_tp  


def load_cmpa_tp(cmpa_path, start_date_str="202005", end_date_str="202005"):
    start_date = datetime.strptime(start_date_str, "%Y%m")
    end_date = datetime.strptime(end_date_str, "%Y%m")
    monthly_dates = pd.date_range(start_date, end_date, freq='MS')
    cmpa_tp = []
    time_idx = []
    daily_idx = []
    for date in monthly_dates:
        # num_days = pd.Timestamp(year, month, 1).days_in_month
        yyyymm = date.strftime("%Y%m")
        print(yyyymm)
        if os.path.exists(f'{cmpa_path}/{yyyymm}'):
            files = sorted(os.listdir(f'{cmpa_path}/{yyyymm}'))
            for file_name in files:
                ds = xr.open_dataset(f"{cmpa_path}/{yyyymm}/{file_name}", engine="zarr",)
                cmpa_tp.append(ds)
                time_idx.append(ds['time'].values)
                daily_idx.append(pd.Timestamp(ds['time'].values[0]).strftime("%Y%m%d"))
    return cmpa_tp, time_idx, daily_idx


def load_static(var="z"):
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    file = f"{project_path}/data/china_{var}.npz"
    data = np.load(file)[var]
    return data


def load_era5_tp(tp_path):
    start_date = datetime.strptime("202403", "%Y%m")
    end_date = datetime.strptime("202408", "%Y%m")
    monthly_dates = pd.date_range(start_date, end_date, freq='MS')
    tp_list = []
   
    for date in monthly_dates:
        # num_days = pd.Timestamp(year, month, 1).days_in_month
        yyyymm = date.strftime("%Y%m")  
        tp = xr.open_dataset(f"{tp_path}/era5_tp_{yyyymm}.zarr", engine="zarr",)
        tp_list.append(tp)
    tp_combined = xr.concat(tp_list, dim="time") 
    return tp_combined


def load_ec_data(ec_dir, start_time, end_time, start_interval_hour):
    start_date = datetime.strptime(start_time, "%Y-%m-%d_%H")
    end_date = datetime.strptime(end_time, "%Y-%m-%d_%H")
    daily_dates = pd.date_range(start_date, end_date, freq=f'{start_interval_hour}h')
    
    surf, high = {}, {}
    for ini_time in daily_dates:
        last_time = ini_time + timedelta(days=5)
        ini_time = ini_time.strftime("%Y%m%d%H")        
        last_time = last_time.strftime("%Y%m%d%H")
        surf_sub = xr.open_dataset(f"{ec_dir}/{ini_time}/ec_sfc_{ini_time}_{last_time}.zarr", 
                                    engine="zarr",)      
        high_sub = xr.open_dataset(f"{ec_dir}/{ini_time}/ec_pl_{ini_time}_{last_time}.zarr", 
                                    engine="zarr",)
        surf[ini_time] = surf_sub
        high[ini_time] = high_sub
    return surf, high


if __name__ == '__main__':
    import arrow
    # load_static("z")
    cmpa_tp, time_idx, daily_ts = load_cmpa_tp("202108", "202108")

    
