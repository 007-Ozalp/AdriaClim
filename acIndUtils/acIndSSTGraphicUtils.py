from datetime import datetime

import xarray as xr
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
import seaborn as sns
rcParams['figure.figsize'] = 18, 10
rcParams['lines.linewidth'] = 3
import csv
import netCDF4
from scipy import stats

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import geocat.datafiles as gdf
from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil
import regionmask


from acIndUtils import acIndUtils


fontSizeLegend = 15
fontSizeTickLabels = 13
fontSizeAxisLabel = 16



def acGenerate1DTendency(t,NcFile1Doutput):    
    
    """ read file 1D fix dimension NetCDF file for the overall SST tendency 
    """

    lon_name = 'lon'
    lat_name = 'lat'
    
    fy_1D= t.mean(dim=(lat_name, lon_name), skipna=True)    
    fy_dt = fy_1D.groupby('time.year').mean()
    df = fy_dt.to_dataframe().reset_index().set_index('year')
    
    x=df.index
    y=df.thetao
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    k=intercept + slope*x
    
    plt.plot(df.thetao, color='teal', marker='o', markerfacecolor='firebrick', markeredgecolor='g', markersize=8)
    plt.plot(x, k, 'k')
    #plt.legend()
    plt.grid()
    plt.title('Annual Trend:Temperature at Sea Surface', size=20)
    plt.ylabel('Temperature (C)',fontsize=15)
    plt.xlabel('TS',fontsize=15)
    plt.xticks(size = 15)
    plt.yticks(size = 15)



def acGenerateDailyTimeSeries(temp_ts):
    
    """ Daily TS SST Analysis 
    """    
    
    file2 = pd.read_csv(temp_ts,index_col='DATE', parse_dates=['DATE'])
    
    plt.title('Temperature at Sea Surface from 1987 to 2019 in the Adriatic Sea', size=20)
    plt.grid()
    plt.plot(file2, color='teal')
    plt.xticks(size = 20)
    plt.yticks(size = 20)
   

        
def acGenerateDailyTimeSeriesSTD(temp_ts):
    
    """ Daily TS STD Analysis 
    """ 
    
    file2 = pd.read_csv(temp_ts,index_col='DATE', parse_dates=True)
    fy_dt = file2.groupby(pd.Grouper(freq='M')).mean()
    
    
    daily_sdT = fy_dt.rolling(window = 12).std()
    
    #plt.plot(daily_sdT, color='k',label='STD', linewidth=3)
    plt.title('STD Temperature at Sea Surface from 1987 to 2019 in the Adriatic Sea', size=20)
    plt.grid()
    plt.plot(daily_sdT, color='teal')
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    return daily_sdT


    
def acGenerateViolinPlotOfDailyData(temp_ts):   

    """ Monthly TS Violin Plot 
    """ 
    
    dateColId = 0
    sstColId = 1
        
    file2 = pd.read_csv(temp_ts)
    file2.iloc[:,dateColId] = pd.to_datetime(file2.iloc[:,dateColId])
    dateCol = file2.iloc[:,dateColId]
    tCol = file2.iloc[:,sstColId]
    file2['Time Series from 1987 to 2019'] = [d.year for d in dateCol]
    file2['month'] = [d.strftime('%b') for d in dateCol]
    years = file2['Time Series from 1987 to 2019'].unique()

    fig, axes = plt.subplots(1, figsize=(18,8), dpi= 100)
    sns.violinplot(x='month', y=tCol.name, data=file2.loc[~file2.month.isin([1987, 2019]), :], palette="tab10", bw=.2, cut=1, linewidth=1)

    axes.set_title('CMEMS Ocean Model Dataset\n Temperature at Sea Surface from 1987 to 2019 in the Adriatic Sea', fontsize=20); 
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.xlabel('TS',fontsize=18)
    plt.ylabel('Sea Surface Temperature (C)',fontsize=18)

    return fig



def acPlotSSTTimeSeries(dailySSTCsv):
    
    dateColId = 0
    sstColId = 1

 #loading and grouping by year/month
    ds = pd.read_csv(dailySSTCsv)
    ds.iloc[:,dateColId] = pd.to_datetime(ds.iloc[:,dateColId])
    dtCol = ds.iloc[:,dateColId]
    sstCol = ds.iloc[:,sstColId]
    mts = ds.groupby([(dtCol.dt.year), (dtCol.dt.month)]).mean()
    
    ssnl = ds.groupby(dtCol.dt.month).mean()
    ssnlvl = ssnl.values
    nmnt = len(mts)
    ssnltl = np.array([ssnlvl[im % 12][0] for im in range(nmnt)]).reshape([nmnt,1])
    anmlMon = mts - ssnltl

 #getting the annual means
    ymean = ds.groupby(dtCol.dt.year).mean() 
    anmlYr = ymean - ymean.mean()

    dtmAnmlMon = [datetime(dt[0], dt[1], 1) for dt in anmlMon.index.values]
    dtmAnmlYr = [datetime(dt, 7, 1) for dt in anmlYr.index.values]

 #getting the trend slope and p-value
    medslope, medintercept, lo_slope, up_slope, pvalue = acIndUtils.acComputeAnnualTheilSenFitFromDailyFile(dailySSTCsv)
    confInt = (up_slope - lo_slope)/2

    f = plt.figure(figsize=[15, 7])
    plt.plot(dtmAnmlMon, anmlMon, color="teal", linewidth=1, label="Monthly anomaly")
    plt.plot(dtmAnmlYr, anmlYr, marker='o', markerfacecolor="firebrick", markeredgecolor='k', linewidth=0, label="Annual anomaly")
    yrii = np.arange(len(dtmAnmlYr))
    trndln = yrii*medslope
    trndln = trndln - np.mean(trndln)
    plt.plot(dtmAnmlYr, trndln, linewidth=.5, color="k", label=f"Trend = {medslope:2.3f}±{confInt:2.3f} K/year")

    plt.grid("on")

    plt.legend(fontsize=fontSizeLegend, loc="upper left", frameon=False)
    plt.xticks(fontsize=fontSizeTickLabels)
    plt.yticks(fontsize=fontSizeTickLabels)

    plt.xlabel("Year", fontsize=fontSizeAxisLabel)
    plt.ylabel("SST anomaly", fontsize=fontSizeAxisLabel)

    plt.tight_layout()

    return f



def plotMeanMap(meanNcFileSpec, areaPerimeter, plotTitle):
    t = xr.open_dataset(meanNcFileSpec.ncFileName)
    temp = t['thetao'][:,:,:]
    temp_av= np.mean(temp[:],axis = 0)
    region_area_1 = regionmask.Regions([areaPerimeter])
    mask_pygeos_area_1 = region_area_1.mask(t.thetao, method="shapely")
    thetao_area_1 = temp_av.values
    thetao_area_1[np.isnan(mask_pygeos_area_1)] = np.nan
    fig = plt.figure(figsize=(12, 12))

    #  coastlines, and adding features
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    ax.coastlines(linewidths=1, alpha=0.9999, resolution="10m")
    
    
    
    
    # Import an NCL colormap
    newcmp = gvcmaps.NCV_jet
    
    vmin = np.nanpercentile(temp_av, .01)
    vmax = np.nanpercentile(temp_av, 99.99)

    # Contourf-plot data: external contour
    heatmap = temp_av.plot.contourf(ax=ax,
                              transform=projection,
                              levels=60,
                              vmin=vmin,
                              vmax=vmax,
                              cmap=newcmp,
                              add_colorbar=False)
    
    
    
    lines=temp_av.plot.contour(ax=ax,alpha=1,linewidths=0.5,colors = 'k',linestyles='None',levels=54)
    
    gvutil.add_major_minor_ticks(ax, y_minor_per_major=1, labelsize=15)
    
    gvutil.set_axes_limits_and_ticks(ax,
                                     xlim=(12, 22),
                                     ylim=(37, 46),
                                     xticks=np.linspace(12, 22, 6),
                                     yticks=np.linspace(37, 46, 10))
    
    
    cbar_ticks=np.arange(5, 30, 1)
    cbar = plt.colorbar(heatmap,
                        orientation='horizontal',
                        shrink=0.8,
                        pad=0.073,
                        extendrect=True,
                        ticks=cbar_ticks)
    
    
    
    ax.set_extent([12, 20.2, 39.7, 46])
    #ax.set_extent([12.1458333333333321, 14.8124999999999982, 43.9583340312159336, 45.7916678850040881])
    #ax.set_extent([12.9030545454545, 15.5260363636364, 42.67, 45.1154181818182])
    #ax.set_extent([14.0166636363636, 17.1778454545455, 41.9038828181818, 43.8950363636364])
    #ax.set_extent([15.8904181818182, 19.9877818181818, 39.6991617292197, 43.1738818181818])
    gvutil.set_titles_and_labels(
        ax,
        maintitle="Annual Mean SST in the Adriatic Sea",
        maintitlefontsize=16,
        xlabel="",
        ylabel="")
   #plt.tight_layout()
    ax.xlabel_style = {'size': 20, 'color': 'k'}
    ax.ylabel_style = {'size': 20, 'color': 'k'}




def plotTrendMap(trendNcFileSpec, areaPerimeter):
    # TODO: revise
    t = xr.open_dataset(trendNcFileSpec.ncFileName)
    region_area_1 = regionmask.Regions([areaPerimeter])

    mask_pygeos_area_1 = region_area_1.mask(t.thetao, method="shapely")

    thetao_area_1 = t.thetao.values
    thetao_area_1[np.isnan(mask_pygeos_area_1)] = np.nan
    fig = plt.figure(figsize=(12, 12))

    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    ax.coastlines(linewidths=1,alpha=0.9999, resolution="10m")
    
    newcmp = gvcmaps.GMT_hot
    reversed_color_map = newcmp.reversed()
    
    heatmap = t.thetao.plot.contourf(ax=ax,
                              transform=projection,
                              levels=50,
                              vmin=0.025,
                              vmax=0.055,
                              cmap=reversed_color_map,
                              add_colorbar=False)
    
    lines=t.thetao.plot.contour(ax=ax,alpha=1,linewidths=0.5,colors = 'k',linestyles='None',levels=50)
    gvutil.set_axes_limits_and_ticks(ax,
                                     xlim=(12, 22),
                                     ylim=(37, 46),
                                     xticks=np.linspace(12, 22, 6),
                                     yticks=np.linspace(37, 46, 10))
    
    gvutil.add_major_minor_ticks(ax, y_minor_per_major=1, labelsize=15)
    
    
    cbar_ticks=np.arange(0.01, 0.07, 0.005)
    cbar = plt.colorbar(heatmap,
                        orientation='horizontal',
                        shrink=0.8,
                        pad=0.073,
                        extendrect=True,
                        ticks=cbar_ticks)
    ax.set_extent([12, 20.2, 39.7, 46])
    
    gvutil.set_titles_and_labels(
        ax,
        maintitle="Trend of SST in the Ariatic Sea",
        maintitlefontsize=16,
        xlabel="",
        ylabel="")
    
    ax.xlabel_style = {'size': 20, 'color': 'k'}
    ax.ylabel_style = {'size': 20, 'color': 'k'}
    
   #plt.tight_layout()





