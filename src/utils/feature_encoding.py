import numpy as np 
import pandas as pd 
import torch 
import math 
import torch.nn.functional as F

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


def get_patch_posi(lats, lons, levels, patch_size=[2, 4, 4]):
    """get the average latitdue / longitude / level of patches

    Args:
        lats (_type_): _description_
        lons (_type_): _description_
        levels (_type_): _description_
        patch_size (list, optional): _description_. Defaults to [2, 4, 4].

    Returns:
    """
    lats = torch.tensor(lats, dtype=torch.float32)
    lons = torch.tensor(lons, dtype=torch.float32)
    levels = torch.tensor(levels, dtype=torch.float32)
 
     # padding
    D, H, W = len(levels), lats.shape[0], lats.shape[1]
    if W % patch_size[2] != 0:
        lats = F.pad(lats, (0, patch_size[2] - W % patch_size[2]))
        lons = F.pad(lons, (0, patch_size[2] - W % patch_size[2]))

    if H % patch_size[1] != 0:
        lats = F.pad(lats, (0, 0, 0, patch_size[1] - H % patch_size[1]))
        lons = F.pad(lons, (0, 0, 0, patch_size[1] - H % patch_size[1]))

    if D % patch_size[0] != 0:
        levels = F.pad(levels, (0, patch_size[0] - D % patch_size[0]))

    # convolution
    kernel_2d = torch.ones((1, 1, patch_size[1], patch_size[2])) / (patch_size[1] * patch_size[2])
    lats = F.conv2d(lats.unsqueeze(0).unsqueeze(0), kernel_2d, stride=(patch_size[1], patch_size[2]))
    lons = F.conv2d(lons.unsqueeze(0).unsqueeze(0), kernel_2d, stride=(patch_size[1], patch_size[2]))
    
    kernel_1d = torch.ones((1, 1, patch_size[0]))/patch_size[0]
    levels = F.conv1d(levels.unsqueeze(0).unsqueeze(0), kernel_1d, stride=patch_size[0])
    return lats.squeeze(0).squeeze(0), lons.squeeze(0).squeeze(0), levels.squeeze(0).squeeze(0)


def get_coords(region="china", degree=True):
    if region == "china":
        lonmin, lonmax = 70, 140
        latmin, latmax = 0, 60
        
        res = 0.25
    lats = np.arange(latmin, latmax + res, res)[::-1]
    lons = np.arange(lonmin, lonmax + res, res)
    coords = np.array(np.meshgrid(lons, lats))
    
    if degree:
        return coords
    else:
        return np.deg2rad(coords)


def get_time_features(t):
    cur_time = pd.to_datetime(t)
    y_h = [cur_time.day_of_year / 366 * 2 * np.pi, cur_time.hour / 24 * 2 * np.pi]
    y_h = np.array(y_h, dtype=np.float32)
    y_h = np.concatenate([np.sin(y_h), np.cos(y_h)], axis=-1)
    time_features = y_h.reshape(-1)
    return time_features


def cal_freqs(v_min, v_max, dim):
    freqs = torch.exp(
                    math.log(v_min) + \
                    torch.arange(0, dim, dtype=torch.float32) * (math.log(v_max) - math.log(v_min)) / (dim -1)
                      )
    return freqs


def time_encoding(t, v_min=1, v_max=366*24, encoding_dim=96):
    """create time encoding
    Args:
        t: a 1-D Tensor of N indices, one per batch element.
        v_min (int, optional): Defaults to 1.
        v_max (_type_, optional): Defaults to 366*24.
        encoding_dim (int, optional): Defaults to 96.
    Returns:
        t_encoding: [N, encoding_dim]
    """
    half = encoding_dim // 2
    freqs = cal_freqs(v_min, v_max, half)
    t_encoding = torch.cat([torch.cos(2 * np.pi * t[:, None].float()/freqs[None]), 
                            torch.sin(2 * np.pi * t[:, None].float()/freqs[None])], dim=-1)
    
    if encoding_dim % 2:
        t_encoding = torch.cat([t_encoding, torch.zeros_like(t_encoding[:, :1])], dim=-1)
    return t_encoding


def position_encoding(lat, lon, v_min=0.01, v_max=720, encoding_dim=96):
    """create time encoding
    Args:
        lat: a 2-D Tensor.
        lon: a 2-D Tensor.
        v_min:
        v_max:
        encoding_dim (int, optional): Defaults to 96.
    Returns:
        p_encoding: [encoding_dim, n_lat, n_lon]
    """   
    half = encoding_dim // 2
    freqs = cal_freqs(v_min, v_max, half)
    p_encoding = torch.cat([torch.cos(2 * np.pi * lat[:, :, None]/freqs[None, None]), 
                            torch.sin(2 * np.pi * lon[:, :, None]/freqs[None, None])], dim=-1)
    
    if encoding_dim % 2:
        p_encoding = torch.cat([p_encoding, torch.zeros_like(p_encoding[:, :1])], dim=-1)
    return p_encoding.permute(2, 0, 1)


def level_encoding(pressure, v_min=0.01, v_max=10000, encoding_dim=96 ):
    """create time encoding
    Args:
        pressure: a 1-D Tensor.
        v_min:
        v_max:
        encoding_dim (int, optional): Defaults to 96.
    Returns:
        p_encoding: [encoding_dim, level]
    """   
    half = encoding_dim // 2
    freqs = cal_freqs(v_min, v_max, half)
    l_encoding = torch.cat([torch.cos(2 * np.pi * pressure[:, None].float()/freqs[None]), 
                            torch.sin(2 * np.pi * pressure[:, None].float()/freqs[None])], dim=-1)
    
    if encoding_dim % 2:
        l_encoding = torch.cat([l_encoding, torch.zeros_like(l_encoding[:, :1])], dim=-1)
    return l_encoding.permute(1, 0)


def get_posi_encoding(region="china", encoding_dim=96, patch_size=[2, 4, 4]):
    lons, lats = get_coords(degree=True)
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    patch_lats, patch_lons, patch_levels = get_patch_posi(lats, lons, levels, patch_size)
    posi_encoding = position_encoding(patch_lats, patch_lons, encoding_dim)
    l_encoding = level_encoding(patch_levels, encoding_dim)
    return posi_encoding, l_encoding
    

if __name__=="__main__":
    coords = get_coords()
    lats, lons, levels = get_patch_posi(coords[1], coords[0], LEVELS, patch_size=[2, 4, 4])
    t = time_encoding(torch.tensor([34, 35]))
    posi = position_encoding(lats, lons)
    level = level_encoding([1000, 900, 850, 500])
