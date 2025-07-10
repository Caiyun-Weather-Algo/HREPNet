import numpy as np
from typing import OrderedDict
import yaml
from matplotlib import cm
import matplotlib.pyplot as plt
import os 

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def color_convert(data):
    cmap = cm.get_cmap('gist_ncar', 1024)
    cmp_min, cmp_max = [-5.22, 214.06]
    data_min, data_max = [0, 100]
    data[data<data_min] = data_min
    data[data>data_max] = data_max

    data = (data - cmp_min)/(cmp_max - cmp_min)
    data = np.log10(data * 90 + 1) / np.log10(91)
    thres = (0.1 - cmp_min) / (cmp_max - cmp_min)
    thres = np.log10(thres * 90 + 1) / np.log10(91)
    show_area = data > thres

    # Convert to color
    rgba = cmap(data)
    rgba[:, :, -1] = rgba[:, :, -1] * 0.4
    rgba *= show_area[:, :, None].astype(float)
    rgba = (rgba * 255).astype(np.uint8)
    return rgba


def near_pad(y, pad=3, thres=0, default=0):
    local_idx = np.indices((pad,pad))
    i0, i1 = np.where(y>thres)
    global_i0 = (i0[:,None,None] - (local_idx[0] - (pad-1)/2)).flatten().astype(int)
    global_i1 = (i1[:,None,None] - (local_idx[1] - (pad-1)/2)).flatten().astype(int)
    global_i0[global_i0<thres] = default
    global_i0[global_i0>=y.shape[0]] = y.shape[0] - 1
    global_i1[global_i1<thres] = default
    global_i1[global_i1>=y.shape[1]] = y.shape[1] - 1
    y2 = np.ones_like(y) * default
    y2[global_i0, global_i1] = y[i0,i1].repeat(pad*pad)
    return y2


def gen_surf_grid(data, lats, lons, BOUNDS=(3.9079, 71.9282, 57.9079, 150.6026), SHAPE=(6000, 7500)):
    condition = np.where(np.logical_and(np.logical_and(lons >= BOUNDS[1], lons <= BOUNDS[-1]), np.logical_and(lats >= BOUNDS[0], lats <= BOUNDS[2])))
    data = data[condition] # There are points outside of the box, remove them
    lons = lons[condition]
    lats = lats[condition]
    # generate griding index
    lats_index = np.around((BOUNDS[2] - lats) / (BOUNDS[2] - BOUNDS[0])*SHAPE[0]).astype(np.int) # lats descending
    lons_index = np.around((lons - BOUNDS[1]) / (BOUNDS[3] - BOUNDS[1])*SHAPE[1]).astype(np.int)
    data_grid = 9999 * np.ones(SHAPE, dtype=np.float32)
    data_grid[lats_index, lons_index] = data
    return data_grid


def lat_weight():
    lats = np.arange(60, 0-0.1, -0.25)
    lat = np.tile(lats, (281, 1)).T
    total = np.cos(lat[:,0]/180*np.pi).sum()
    w = lat.shape[0] * np.cos(lat/180*np.pi) / total
    return w


def get_interp_idxs(src_shape=(241, 281), dst_shape=(256, 256)):
    facs = np.array(dst_shape, dtype=np.float16)/np.array(src_shape, dtype=np.float16)
    idxs = np.indices(dst_shape)
    row_idxs = np.round(idxs[0][:,0]/facs[0]).astype(int)
    col_idxs = np.round(idxs[1][0,:]/facs[1]).astype(int)
    return (row_idxs, col_idxs)


def plot_score(score, x, xlabel, ylabel, title, figpath):
    plt.figure(dpi=300, figsize=(6,8))
    plt.plot(x, score, "ro-", linewidth=1.5)
    plt.xlim([0, x[-1]])
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
    plt.title(title, fontsize=15)
    plt.savefig(figpath)
    plt.close()
    return


def get_era5_stats():
    variables = OrderedDict(yaml.load(open(f"{base_path}/config/era5_variables.yaml", "r"), Loader=yaml.Loader))
    variables["levels"] = list(range(100, 1000+1, 50)) + [925, 50]
    
    surf_limits = np.load(f"{base_path}/data/surf_var_limit_grid_5ys.npz")["x"]  ##TODO change limit
    high_limits = np.load(f"{base_path}/data/high_var_limit_grid_5ys.npz")["x"]
    surf_meanstd = np.load(f"{base_path}/data/surf_var_mean_std_38ys.npz")["x"]
    high_meanstd = np.load(f"{base_path}/data/high_var_mean_std_38ys.npz")["x"] # t, sp, z, u, v
    cloud_meanstd = np.load(f"{base_path}/data/high_cloud_mean_max_5ys.npz")["x"]
    high_meanstd_925_50 = np.load(f"{base_path}/data/high_925_50hPa_mean_std_38ys.npz")["x"] # z, t, sp, u, v
    
    for s, v in zip(("minmax", "meanstd"), ([surf_limits, high_limits], [surf_meanstd, high_meanstd])):
        variables["surface"]["2mt"][s] = v[0][:, 0]
        variables["surface"]["mslp"][s] = v[0][:, 1]
        variables["surface"]["10m_u_component_of_wind"][s] = v[0][:, 2]
        variables["surface"]["10m_v_component_of_wind"][s] = v[0][:, 3]
        # high level
        if s == "minmax":
            variables["high"]["temperature"][s] = v[1][:, 0]
            variables["high"]["specific_humidity"][s] = v[1][:, 1]
            variables["high"]["geopotential"][s] = v[1][:, 2]
            variables["high"]["u_component_of_wind"][s] = v[1][:, 3]
            variables["high"]["v_component_of_wind"][s] = v[1][:, 4]
        else:
            variables["high"]["temperature"][s] = np.concatenate((v[1][:, 0], high_meanstd_925_50[:, 1]), axis=1)
            variables["high"]["specific_humidity"][s] = np.concatenate((v[1][:, 1], high_meanstd_925_50[:, 2]), axis=1)
            variables["high"]["geopotential"][s] = np.concatenate((v[1][:, 2], high_meanstd_925_50[:, 0]), axis=1)
            variables["high"]["u_component_of_wind"][s] = np.concatenate((v[1][:, 3], high_meanstd_925_50[:, 3]), axis=1)
            variables["high"]["v_component_of_wind"][s] = np.concatenate((v[1][:, 4], high_meanstd_925_50[:, 4]), axis=1)

    # cloud
    for stats, value in zip(["meanstd"], [cloud_meanstd]):
        variables['high']['qcloud'][stats] = value[:, 0]
        variables['high']['qice'][stats] = value[:, 1]

    # tp
    variables["surface"]["total_precipitation"] = {}
    variables["surface"]["total_precipitation"]["minmax"] = [100, 0]
    variables["surface"]["total_precipitation"]["meanstd"] = [0, 42.40353617054452]  #dbz [0.0025163612293158523, 0.028196929403035864]
    # cmpa
    variables["surface"]["cmpa_precipitation"] = {}
    variables["surface"]["cmpa_precipitation"]["meanstd"] = [0, 46.29922575990931] #dbz
    variables["surface"]["station_precipitation"] = {}
    variables["surface"]["station_precipitation"]["minmax"] = [100, 0]
    variables["surface"]["station_precipitation"]["meanstd"] = [0.0025163612293158523, 0.028196929403035864]
    return variables

            
def nearest_interp_idxs(src_shape=(241, 281), dst_shape=(256, 256)):
    facs = np.array(dst_shape, dtype=np.float16)/np.array(src_shape, dtype=np.float16)
    idxs = np.indices(dst_shape)
    row_idxs = np.round(idxs[0][:,0]/facs[0]).astype(int)
    col_idxs = np.round(idxs[1][0,:]/facs[1]).astype(int)
    return (row_idxs, col_idxs)


def resize(data, idxs):
    if len(data.shape)==3:
        interped = data[:, idxs[0], :][:, :, idxs[1]]
    elif len(data.shape)==4:
        interped = data[:, :, idxs[0], :][:,:, :, idxs[1]]
    elif len(data.shape)==2:
        interped = data[idxs[0], :][:, idxs[1]]
    return interped


def static4eval(era5_test_loader):
    surf_avg = np.load("./share/surf_var_mean_pix-lev_38ys.npz")["data"]  # for ACC calculation
    high_avg = np.load("./share/high_var_mean_pix-lev_38ys.npz")["data"]  # for ACC calculation
    #surf_avg = np.load("./share/surf_var_mean_std_38ys.npz")["x"][0]  # for ACC calculation
    #high_avg = np.load("./share/high_var_mean_std_38ys.npz")["x"][0]  # for ACC calculation
    high_avg = np.concatenate((high_avg[2:3], high_avg[:1], high_avg[1:2], high_avg[3:]))
    high_avg = high_avg[:, era5_test_loader.input_lev_idxs]
    return surf_avg, high_avg


def tp2dbz(x, dmin=2, dmax=55):
    x[x < 0.04] = 0.04
    y = 10 * np.log10(200 * np.power(x, 1.6))

    y[y < dmin] = dmin    # 2 dbz ~ 0.05 mm/h
    y[y > dmax] = dmax    # 55 dbz ~ 100 mm/h
    y = y #/ 55
    return y


def dbz2tp(dbz):
    dbz = dbz #* 55
    tp = np.power(10 ** (dbz / 10) / 200, 1 / 1.6)
    tp[tp < 0.1 ] = 0
    return tp


if __name__ == '__main__':
    stats = get_era5_stats()
    import pickle
    with open('./data/era5_stats_1980-2018.pkl', 'wb') as pickle_file:
        pickle.dump(stats, pickle_file)
    # with open('./data/era5_stats_1980-2018.pkl', 'rb') as pickle_file:
    #     loaded_data = pickle.load(pickle_file)
    # print(loaded_data)
    