import numpy as np 
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm


ColorBar_CONFIG = {
    "rainfall": {
        "levels": [0, 0.1, 1, 2, 5, 10, 15, 25, 50],       
        "colors": ['#FFFFFF', '#A6F28F', '#77D970', '#3FB445','#61BBFF','#0000FF','#FA00FA','#800040']
            },
    "rainfall_cmpa": {
        "levels": [0, 0.1, 0.25, 0.5, 1,
                   2, 3, 5, 7, 10, 
                    15, 20, 25, 30],       
        "colors": ['#FFFFFF', '#61BBFF', '#3FB445', '#77D970', '#A6F28F',
                   '#c0fa8b','#D7FC94', '#F4FE9E', '#faee66', '#F4CD00',
                   '#ED9F00','#E67301', '#DD4900', '#D52300', '#CD0000',
                   ]
    },  #
    "rainfall_ct": {
        "levels": [0, 0.1, 0.5, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30],  # np.arange(0, 50.01, 1.25),
        "colors": cm.get_cmap('BuPu')(np.linspace(0, 1, 16))[-15:]
        
    },  # rainfall colorbar use cmpa 
    
    "rainfall_r": {
        "levels": [0,  0.1,  0.2, 0.5,  0.75,  1.25, 
                   1.5,  2,  2.5, 3, 4.5, 
                   6, 8, 10, 12, 14, 
                   16, 18, 20, 25, 50,
                   100],
        "colors": ['#FFFFFF', '#ADD8E6', '#1D90FF', '#02FFFF', '#1EFF1D', 
                   '#04CB02', '#009700', '#006400', '#FFFF00', '#F4CD00', 
                   '#ED9F00', '#E67301', '#DD4900', '#D52300', '#CD0000',
                   '#FF00FF', '#D10aE7', '#AB10D0', '#8A14B9', '#6D18A2',
                   '#54198B']
    },  # rainfall colorbar similar to radar
    "metric": {
        "levels": np.arange(0, 1, 0.05), # np.arange(0, 50.01, 1.25),
        "colors": cm.get_cmap('BuPu')(np.linspace(0, 1, 21))
        
    },  # rainfall colorbar used for cmpa dbz
    "radar":{
        "levels": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        "colors": ['#FFFFFF', '#01A0F6', '#00ECEC', '#01FF00', '#00C800', '#019000', '#FFFF00', '#E7C000', '#FF9000', '#FF0000', '#D60000', '#C00000', '#FF00F0', '#780084', '#AD90F0']
    },
    "radar_diff":{
        "levels": np.arange(-5, 5.01, 0.25),
        "colors": cm.get_cmap('RdBu')(np.linspace(0, 1, 41))[::-1]
    },    
    "radar_diff_norm":{
        "levels": np.arange(-1, 1.01, 0.05),
        "colors": cm.get_cmap('RdBu')(np.linspace(0, 1, 41))[::-1]
    }     
}


def define_map(region="era5"):
    if region == "era50":
        lonmin, lonmax = 70, 140
        latmin, latmax = 0, 60
        extents = [lonmin, lonmax, latmin, latmax]
        
        res = 0.25
    if region == "era5":
        lonmin, lonmax = 70, 140
        latmin, latmax = 0, 60
        extents = [lonmin, lonmax, 15, latmax]
        
        res = 0.25
    if region == "cmpa":
        lonmin, lonmax = 70.00, 139.99
        latmin, latmax = 15.00, 59.99
        extents = [lonmin, lonmax, latmin, latmax]
        
        res = 0.01
    if region == "cmpa_5km":
        lonmin, lonmax = 70.025, 139.975
        latmin, latmax = 15.025, 59.975
        extents = [lonmin, lonmax, latmin, latmax]
        
        res = 0.05
    lats = np.arange(latmin, latmax + res, res)
    lons = np.arange(lonmin, lonmax + res, res)
    return extents, lats, lons 


def plot(fig_name, data, levels):
    extents, lats, lons = define_map()
    fig = plt.figure()
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(extents, crs=proj)

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)
    # 画出填色图.
    im = ax.contourf(
        lons, lats, data,
        levels=levels, cmap='YlGnBu',
        extend='both', alpha=0.8
    )
    cbar = fig.colorbar(
        im, ax=ax, shrink=0.9, pad=0.1, orientation='horizontal',
        # format=mticker.PercentFormatter()
    )
    cbar.ax.tick_params(labelsize='small')
    plt.savefig(fig_name)


def plot_contour_map(ax, fig, extents, proj, lons, lats, data, levels=None, colors=None, cbar=True):
    """
    Plot contour map
    """
    ax.set_extent(extents, crs=proj)
    coastline = cfeature.NaturalEarthFeature(category='physical', name='coastline',
                                             scale='110m', facecolor='none')
    ax.add_feature(coastline, edgecolor='black')
    # borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land',
    #                                    scale='110m', facecolor='none')
    # ax.add_feature(borders, edgecolor='black', linestyle=':') 
    im = ax.contourf(
        lons, lats, data,
        levels=levels, colors=colors, extend='both', alpha=1
    )
    if cbar:
        cbar = fig.colorbar(
            im, ax=ax, shrink=0.4, pad=0.1, orientation='vertical',
            # format=mticker.PercentFormatter()
        )
        cbar.ax.tick_params(labelsize='small')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False    
    gl.right_labels = False  
    gl.xlabel_style = {'size': 10, 'color': 'black'}  
    gl.ylabel_style = {'size': 10, 'color': 'black'}  

    ax.set_title('target')
    return im 
    
 
def plot_tp_result(fig_name, attr, output, target):
    extents, lats, lons = define_map(region='era5')
    cmpa_extents, cmpa_lats, cmpa_lons = define_map(region='cmpa')

    colors = ColorBar_CONFIG[attr]["colors"]
    levels = ColorBar_CONFIG[attr]["levels"]
       
    fig = plt.figure(figsize=(9, 5))
    proj = ccrs.PlateCarree()
    # target
    ax = fig.add_subplot(1, 2, 1, projection=proj)
    plot_contour_map(ax, fig, extents, proj, cmpa_lons, cmpa_lats, target, levels, colors)
    ax.set_title('target')
    # predict
    ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    plot_contour_map(ax2, fig, extents, proj, cmpa_lons, cmpa_lats, output, levels, colors)
    ax2.set_title('predict')

    plt.savefig(fig_name + '.png')
    plt.close()
    

def plot_single_var(fig_name, target, region='cmpa'):
    extents, lats, lons = define_map(region=region)
    
    attr = "rainfall_cmpa"
    colors = ColorBar_CONFIG[attr]["colors"]
    levels = ColorBar_CONFIG[attr]["levels"]
    
    fig = plt.figure(figsize=(9, 5))
    proj = ccrs.PlateCarree()
    # target
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    plot_contour_map(ax, fig, extents, proj, lons, lats, target, levels, colors)
    ax.set_title('target')

    plt.savefig(fig_name + '.png')
    plt.close()


def plot_single_var_3hour(fig_name, title, lead_hour, target, region='cmpa'):
    extents, lats, lons = define_map(region=region)
       
    fig, axes  = plt.subplots(nrows=4, ncols=6, figsize=(18, 12), 
                              subplot_kw={'projection': ccrs.PlateCarree()},
                              constrained_layout=True)
    proj = ccrs.PlateCarree()
    axes = axes.flatten() 
    # target
    i = 0
    for ax in axes:    
        im = plot_contour_map(ax, fig, extents, proj, lons, lats, target[i], cbar=False)
        if i%3 == 0:
            ax.set_title(f"EC_T+{i+lead_hour}")
        else:
            ax.set_title(f"Pangu_T+{i+lead_hour}")

        i += 1
        
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.04, shrink=0.45)
    cbar.ax.tick_params(labelsize='small')
    # 自动调整子图之间的间距，避免标签和子图重叠
    plt.suptitle(title)
    plt.savefig(fig_name + '.png', bbox_inches='tight')
    plt.close()
     
 
def plot_single_var_24hour(fig_name, title, lead_hour, target, region='cmpa'):
    extents, lats, lons = define_map(region=region)

    attr = "rainfall_cmpa"
    colors = ColorBar_CONFIG[attr]["colors"]
    levels = ColorBar_CONFIG[attr]["levels"]
       
    fig, axes  = plt.subplots(nrows=4, ncols=6, figsize=(18, 12), 
                              subplot_kw={'projection': ccrs.PlateCarree()},
                              constrained_layout=True)
    proj = ccrs.PlateCarree()
    axes = axes.flatten() 
    # target
    for i in range(target.shape[0]):  
        ax = axes[i]  
        im = plot_contour_map(ax, fig, extents, proj, lons, lats, target[i], levels, colors, cbar=False)
        if i%3 == 0:
            ax.set_title(f"CMPA_T+{i+lead_hour}")
        else:
            ax.set_title(f"CMPA_T+{i+lead_hour}")
        
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.04, shrink=0.45)
    cbar.ax.tick_params(labelsize='small')
    # 自动调整子图之间的间距，避免标签和子图重叠
    plt.suptitle(title)
    plt.savefig(fig_name + '.png', bbox_inches='tight')
    plt.close()
    
      
def plot_result_womap(fig_name, output, target):
   
    fig = plt.figure(figsize=(9, 5))
    # target
    ax = fig.add_subplot(1, 2, 1)
    contour = plt.contourf(target)
    plt.colorbar(contour)     
    ax.set_title('target')
    # predict
    ax2 = fig.add_subplot(1, 2, 2)
    contour = plt.contourf(output)
    plt.colorbar(contour)    
    ax2.set_title('predict')

    plt.savefig(fig_name + '.png')
    plt.close()
    
             
def plot_train_result(fig_name, attr, output, target, region='cmpa', store=True, label=''):
    extents, lats, lons = define_map(region=region)
    
    colors = ColorBar_CONFIG[attr]["colors"]
    levels = ColorBar_CONFIG[attr]["levels"]
       
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)  # 启用 constrained_layout
    proj = ccrs.PlateCarree()
    
    # target
    ax1 = fig.add_subplot(1, 2, 1, projection=proj)
    im1 = plot_contour_map(ax1, fig, extents, proj, lons, lats, target, levels, colors, cbar=False)
    ax1.set_title('Target')
    
    # predict
    ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    im2 = plot_contour_map(ax2, fig, extents, proj, lons, lats, output, levels, colors, cbar=False)
    ax2.set_title('Prediction')
    
    # Adjust and add colorbar
#     position = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical', pad=0.02, shrink=0.65)

    cbar.ax.tick_params(labelsize='small')
#     plt.tight_layout()
    if store:
        plt.savefig(fig_name + '.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_era5_tp(fig_name, attr, init_times, fcst_times, target, region='cmpa', store=True, label=''):
    extents, lats, lons = define_map(region=region)
    
    colors = ColorBar_CONFIG[attr]["colors"]
    levels = ColorBar_CONFIG[attr]["levels"]
       
    fig, axes  = plt.subplots(nrows=4, ncols=6, figsize=(18, 12), 
                              subplot_kw={'projection': ccrs.PlateCarree()},
                              constrained_layout=True)
    proj = ccrs.PlateCarree()
    axes = axes.flatten() 
    print(target.shape)
    i = 0
    for ax in axes: 
        im = plot_contour_map(ax, fig, extents, proj, lons, lats, target[i], levels, colors, cbar=False)
        ax.set_title(f"Init{init_times[i]}_ft{fcst_times[i]}")
        i += 1

    # Adjust and add colorbar
#     position = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', pad=0.02, shrink=0.45)

    cbar.ax.tick_params(labelsize='small')
    plt.suptitle(f"DTM_Model_TP")
        
    if store:
        plt.savefig(fig_name + '.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        

def plot_pred_tp(fig_name, attr, init_time, fcst_time, dtm_pred, ens_members, region='cmpa', store=True, label=''):
    extents_era5, lats_era5, lons_era5 = define_map(region="era5")
    extents_cmpa, lats_cmpa, lons_cmpa = define_map(region="cmpa_5km")
    colors = ColorBar_CONFIG[attr]["colors"]
    levels = ColorBar_CONFIG[attr]["levels"]
       
    fig, axes  = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), 
                              subplot_kw={'projection': ccrs.PlateCarree()},
                              constrained_layout=True)
    proj = ccrs.PlateCarree()
    axes = axes.flatten() 
    im = plot_contour_map(axes[0], fig, extents_era5, proj, lons_era5, lats_era5, dtm_pred, levels, colors, cbar=False)
    axes[0].set_title(f"DTM Model TP")
    
    i = 0
    for member in ens_members: 
        im = plot_contour_map(axes[i+1], fig, extents_cmpa, proj, lons_cmpa, lats_cmpa, member, levels, colors, cbar=False)
        axes[i+1].set_title(f"Mebmber {i+1}")
        i += 1

    # Adjust and add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', pad=0.02, shrink=0.35)

    cbar.ax.tick_params(labelsize='small')
    plt.suptitle(f"Init{init_time}_ft{fcst_time}")
        
    if store:
        plt.savefig(fig_name + '.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        

def plot_pred_residual(fig_name, attr, init_time, fcst_time, dtm_pred, ens_members, region='cmpa', store=True, label=''):
    extents_era5, lats_era5, lons_era5 = define_map(region="era5")
    extents_cmpa, lats_cmpa, lons_cmpa = define_map(region="cmpa_5km")
    colors = ColorBar_CONFIG[attr]["colors"]
    levels = ColorBar_CONFIG[attr]["levels"]
       
    fig, axes  = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), 
                              subplot_kw={'projection': ccrs.PlateCarree()},
                              constrained_layout=True)
    proj = ccrs.PlateCarree()
    axes = axes.flatten() 
    im = plot_contour_map(axes[0], fig, extents_era5, proj, lons_era5, lats_era5, dtm_pred, levels, colors, cbar=False)
    axes[0].set_title(f"DTM Model TP")
    
    i = 0
    attr = "radar_diff_norm"
    colors = ColorBar_CONFIG[attr]["colors"]
    levels = ColorBar_CONFIG[attr]["levels"]
    for member in ens_members: 
        im = plot_contour_map(axes[i+1], fig, extents_cmpa, proj, lons_cmpa, lats_cmpa, member, levels, colors, cbar=False)
        axes[i+1].set_title(f"Mebmber {i+1}")
        i += 1

    # Adjust and add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', pad=0.02, shrink=0.35)

    cbar.ax.tick_params(labelsize='small')
    plt.suptitle(f"Init{init_time}_ft{fcst_time}")
        
    if store:
        plt.savefig(fig_name + '.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
                    
def plot_cloud(fig_name, tp, qv, qc, qi, pressure=1000, level_ratio=1):
    extents, lats, lons = define_map()
    fig = plt.figure(figsize=(10, 10))
    proj = ccrs.PlateCarree()
    # tp  
    colors=['#FFFFFF', '#A6F28F','#3DBA3D','#61BBFF','#0000FF','#FA00FA','#800040']
    levels = np.array([0, 0.001, 0.1, 1, 2, 5, 10, 15])
    ax = fig.add_subplot(2, 2, 1, projection=proj)
    plot_contour_map(ax, fig, extents, proj, lons, lats, tp, levels, colors)
    ax.set_title('TP')
    # qv
    colors = ColorBar_CONFIG["radar"]["colors"]
    levels = np.array([0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2])
    levels_qv = np.array([0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2])
    ax2 = fig.add_subplot(2, 2, 2, projection=proj)
    print("qv", qv.min(), qv.max())
    plot_contour_map(ax2, fig, extents, proj, lons, lats, qv, levels_qv*level_ratio*10, colors)
    ax2.set_title(f'QV({pressure}hPa)')
    # qc
    ax2 = fig.add_subplot(2, 2, 3, projection=proj)
    print("qc", qc.min(), qc.max())
    plot_contour_map(ax2, fig, extents, proj, lons, lats, qc, levels*level_ratio, colors)
    ax2.set_title(f'QC({pressure}hPa)')
    # qi
    ax2 = fig.add_subplot(2, 2, 4, projection=proj)
    print("qi", qi.min(), qi.max())
    plot_contour_map(ax2, fig, extents, proj, lons, lats, qi, levels*level_ratio*0.1, colors)
    ax2.set_title(f'QI({pressure}hPa)')
    
    plt.savefig(fig_name + '.png')
    plt.close()
