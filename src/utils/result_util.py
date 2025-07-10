import os 
import numpy as np
import pandas as pd
import xarray as xr
import arrow
import time


def data_to_npz(data, ini_time, fcst_time, result_path):
    t0 = time.time()
    os.makedirs(f"{result_path}/{ini_time}", exist_ok=True)
    filename = f"{result_path}/{ini_time}/dtm_pred_{ini_time}_{fcst_time}.npz"
    np.savez_compressed(filename, prcp=data)
    # print(f'save data at {time0} cost {time.time() - t0}')


def data_to_zarr(data, timestamp, result_path):
    t0 = time.time()
    time0 = arrow.get(int(timestamp)).format("YYYYMMDDHH")
    times = np.array([pd.to_datetime(time0, format='%Y%m%d%H')])
    lonmin, lonmax = 70, 140
    latmin, latmax = 0, 60        
    res = 0.25
    lat = np.arange(latmin, latmax + res, res)[::-1]
    lon = np.arange(lonmin, lonmax + res, res)

    ds = xr.Dataset(
        {
            "precipitation": (["time", "lat", "lon"], data)
        },
        coords={
            "time": times,
            "lat": lat,
            "lon": lon
        }
    )
    os.makedirs(f'{result_path}/{time0[0:4]}', exist_ok=True)
    filename = f"{result_path}/{time0[0:4]}/prcp_data_{time0}.zarr"
    ds.to_zarr(filename)
    print(f'save data at {time0} cost {time.time() - t0}')
