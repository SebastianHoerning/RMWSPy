
import os
import numpy as np
import xarray as xr
import matplotlib.pylab as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from netCDF4 import Dataset
import bresenhamline as bresenhamline
import matplotlib.ticker as ticker
import pandas as pd



def plot_RG(ax, p_data):
    lats = p_data[:,0]
    lons = p_data[:,1]
    for lon, lat in zip(lons, lats):
        ax.plot(lon, lat, 'o', color='black', markersize=5, zorder=5, transform=ccrs.PlateCarree())
        
def plot_CML(ax, mwl_data):
    lats1 = mwl_data[:,0]
    lons1 = mwl_data[:,1]
    lats2 = mwl_data[:,2]
    lons2 = mwl_data[:,3]
    for lon1, lat1, lon2, lat2 in zip(lons1, lats1, lons2, lats2):
        ax.plot([lon1, lon2], [lat1, lat2], '-', color='black', lw=1, zorder=4, transform=ccrs.PlateCarree())

def plot_pp(datafolder, p_data, mwl_data, fields, mwl_prec, boxlist, x, count_size, tstep, rix):
    # load VR prec
    ncfile = os.path.join(datafolder, r'VR1_P_hour_2015_08.nc')
    nc = Dataset(ncfile)
    nc = xr.open_dataset(ncfile)
    prec_VR = nc.TOT_PREC[tstep,:,:]
            
    # Define colors
    nws_precip_colors = [
                    "#FFFFFF",
                    "#04e9e7",  
                    "#019ff4",  
                    "#0300f4",  
                    "#02fd02",  
                    "#01c501",  
                    "#008e00",  
                    "#fdf802",  
                    "#e5bc00",  
                    "#fd9500",  
                    "#fd0000",  
                    "#d40000",  
                    "#bc0000",  
                    "#f800fd",  
                    "#9854c6"  
                    ]
                    
    clevs = [1, 1.5, 2.0, 2.5, 3, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7., 7.5]

    # create xarray
    level = np.arange((count_size))
    time  = prec_VR.time
    lon   = prec_VR.lon
    lat   = prec_VR.lat

    # create time stamp for figure names
    tstamp = pd.to_datetime(str(time.values)).strftime('%Y_%m_%d-%Hh%Mm%Ss')

    data_RF = xr.DataArray(fields, 
                           dims=['level', 'lat', 'lon'], 
                           coords={'level':level, 'lat':lat, 'lon':lon, 'time':time})
   
    # Define projection for Neckar
    map_proj = ccrs.PlateCarree(central_longitude=8.94)

    percentilRF = np.percentile(data_RF, 90, axis=0)-np.percentile(data_RF, 10, axis=0)
    data_P90_P10_RF = xr.DataArray(percentilRF, 
                           dims=['lat', 'lon'], 
                           coords={'lat':lat, 'lon':lon, 'time':time})

    # plot P90-P10
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection=map_proj)
    plot_RG(ax, p_data)
    plot_CML(ax, mwl_data)
    data_P90_P10_RF.plot(robust=True, ax=ax,
                levels=clevs, colors=nws_precip_colors, cbar_kwargs={'label':'P [mm/h]'},
                transform=ccrs.PlateCarree())
    ax.set_extent([8, 10.37, 47.8, 49.6], crs=ccrs.PlateCarree())
    ax.set_xticks([8, 8.5, 9, 9.5, 10], minor=False, crs=ccrs.PlateCarree())
    ax.set_yticks([48, 48.5, 49, 49.5], minor=False, crs=ccrs.PlateCarree())
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    fig.savefig(r'PC90-PC10_%s.pdf'%tstamp, bbox_inches='tight')
    plt.clf()
    plt.close()


    # plot a single random field, mean of all random fields, and VR-Prec
    fig=plt.figure(figsize=(15,7))   
    ax1=fig.add_subplot(131, projection=map_proj)
    plot_RG(ax1, p_data)
    plot_CML(ax1, mwl_data)
    p = data_RF[rix,:,:].plot(robust=True, ax=ax1,
                levels=clevs, colors=nws_precip_colors, add_colorbar=False,
                transform=ccrs.PlateCarree())
    ax1.set_extent([8, 10.37, 47.8, 49.6], crs=ccrs.PlateCarree())
    ax1.set_xticks([8, 8.5, 9, 9.5, 10], minor=False, crs=ccrs.PlateCarree())
    ax1.set_yticks([48, 48.5, 49, 49.5], minor=False, crs=ccrs.PlateCarree())
    ax1.xaxis.grid(True, which='major')
    ax1.yaxis.grid(True, which='major')
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    t = ax1.set_title('(a)')

    ax2=fig.add_subplot(132, projection=map_proj)
    plot_RG(ax2, p_data)
    plot_CML(ax2, mwl_data)
    data_RF.mean(axis=0).plot(robust=True, ax=ax2,
                levels=clevs, colors=nws_precip_colors, add_colorbar=False, 
                transform=ccrs.PlateCarree())
    ax2.set_extent([8, 10.37, 47.8, 49.6], crs=ccrs.PlateCarree())
    ax2.set_xticks([8, 8.5, 9, 9.5, 10], minor=False, crs=ccrs.PlateCarree())
    ax2.set_yticks([48, 48.5, 49, 49.5], minor=False, crs=ccrs.PlateCarree())
    ax2.xaxis.grid(True, which='major')
    ax2.yaxis.grid(True, which='major')
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    plt.setp(ax2.get_yticklabels(), visible=False)
    t = ax2.set_title('(b)')

    ax3=fig.add_subplot(133, projection=map_proj)
    plot_RG(ax3, p_data)
    plot_CML(ax3, mwl_data)
    prec_VR.plot(robust=True, ax=ax3,
                levels=clevs, colors=nws_precip_colors, add_colorbar=False, 
                transform=ccrs.PlateCarree())
    ax3.set_extent([8, 10.37, 47.8, 49.6], crs=ccrs.PlateCarree())
    ax3.set_xticks([8, 8.5, 9, 9.5, 10], minor=False, crs=ccrs.PlateCarree())
    ax3.set_yticks([48, 48.5, 49, 49.5], minor=False, crs=ccrs.PlateCarree())
    ax3.xaxis.grid(True, which='major')
    ax3.yaxis.grid(True, which='major')
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    plt.setp(ax3.get_yticklabels(), visible=False)
    t = ax3.set_title('(c)')

    cbar = fig.colorbar(p, ax=[ax1, ax2, ax3], shrink=0.6, location='bottom')
    cbar.set_label('P [mm/h]')
    plt.subplots_adjust(wspace=0.05, bottom=0.3)              
    fig.savefig(r'RF_MeanF_VR_%s.pdf'%tstamp, bbox_inches='tight')
    plt.clf()
    plt.close()

    # plot standard deviation
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection=map_proj)
    plot_RG(ax, p_data)
    plot_CML(ax, mwl_data)
    data_RF.std(axis=0).plot(robust=True, ax=ax,
                cmap=plt.cm.Reds, cbar_kwargs={'label':'P [mm/h]'},
                transform=ccrs.PlateCarree())
    ax.set_extent([8, 10.37, 47.8, 49.6], crs=ccrs.PlateCarree())
    ax.set_xticks([8, 8.5, 9, 9.5, 10], minor=False, crs=ccrs.PlateCarree())
    ax.set_yticks([48, 48.5, 49, 49.5], minor=False, crs=ccrs.PlateCarree())
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    fig.savefig(r'Std_%s.pdf'%tstamp, bbox_inches='tight')
    plt.clf()
    plt.close()

    # plot box-plot
    tick_space = 5
    fig1 = plt.figure(1, figsize=(6,12))
    axb = fig1.add_subplot(111)
    bx = axb.boxplot(boxlist, 0, '', 
                    whiskerprops=dict(color='white',linewidth=.0001), \
                    capprops=dict(color='black', linewidth=2.), \
                    medianprops = dict(linewidth=2.), vert=False)
    axb.set_ylabel('CML')
    axb.set_xlabel('P [mm/h]')
    axb.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    axb.yaxis.set_major_locator(ticker.MultipleLocator(tick_space))
    axb.set_yticklabels(np.arange(0,100,5))
    axb.plot( mwl_prec, x, 'x', c='red')
    fig1.savefig(r'Boxplot_CMLobs_vs_CMLRF_%s.pdf'%tstamp, bbox_inches='tight')



