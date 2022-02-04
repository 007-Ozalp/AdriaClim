
import xarray as xr
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse 
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
rcParams['figure.figsize'] = 18, 10
rcParams['lines.linewidth'] = 3
import csv
import netCDF4
from scipy import stats


import adriaClimIndUtils_aggregates as pv 


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
    
    plt.plot(df.thetao, color='red', marker='o', markerfacecolor='lime', markeredgecolor='g', markersize=8)
    plt.plot(x, k, 'blue')
    plt.legend()
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
    plt.plot(file2, color='r')
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
    plt.plot(daily_sdT, color='r')
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    return daily_sdT
    
def acGenerateDailyTimeSeriesPLOT(temp_ts):   

    """ Monthly TS Violin Plot 
    """ 
        
    file2 = pd.read_csv(temp_ts, parse_dates=['DATE'])
    file2['Time Series from 1987 to 2019'] = [d.year for d in file2.DATE]
    file2['month'] = [d.strftime('%b') for d in file2.DATE]
    years = file2['Time Series from 1987 to 2019'].unique()
    plt.style.use('seaborn')

    file2['Time Series from 1987 to 2019'] = [d.year for d in file2.DATE]
    file2['Month'] = [d.strftime('%b') for d in file2.DATE]
    years = file2['Time Series from 1987 to 2019'].unique()

    fig, axes = plt.subplots(1, figsize=(18,8), dpi= 100)
    sns.violinplot(x='Month', y='TEMPERATURE', data=file2.loc[~file2.month.isin([1987, 2019]), :], palette="tab10", bw=.2, cut=1, linewidth=1)

    axes.set_title('CMEMS Ocean Model Dataset\n Temperature at Sea Surface from 1987 to 2019 in the Adriatic Sea', fontsize=20); 
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.xlabel('TS',fontsize=18)
    plt.ylabel('Sea Surface Temperature (C)',fontsize=18)