
import xarray as xr

import pandas as pd

############################  NC MD  ###############################################

def acClipDataOnRegion(dataInputNC,areaPerimeter,dataOutput):

    print("CMEMS SST Dimension:",dataInputNC)
    print("Clipped Area Dimensions:",areaPerimeter)
    
    lat_max=areaPerimeter['LAT'].max()
    lat_min=areaPerimeter['LAT'].min()
    lon_max=areaPerimeter['LON'].max()
    lon_min=areaPerimeter['LON'].min()
    
    #print("Area Extension:",lat_max, lat_min,lon_max, lon_min)
    
    t = dataInputNC.sel(lat=slice(lat_min,lat_max), lon=slice(lon_min,lon_max))

    print("Reseized Area:",t)

    print ('saving to ', dataOutput)
    t.to_netcdf(path=dataOutput)
    print ('finished saving')
    
    return t


############################  NC MD ###############################################

def acGenerate2DAnnualMaps(t,annualMapsNcFile): 
    
    fy_dt = t.groupby('time.year').mean()
    
    print ('annual mean ', fy_dt)
    
    print ('saving to ', annualMapsNcFile)
    fy_dt.to_netcdf(path=annualMapsNcFile)
    print ('finished saving')
    
    return fy_dt

##############################################################################################
##############################################################################################
############################  Winter Mean JFMA ###############################################
############################  Summer Mean JASO ###############################################
############################  MAP ###############################################
############################  MAP ###############################################
##############################################################################################
##############################################################################################

############################  NC MD MEAN DIM ###############################################

import csv
import netCDF4

def acGenerate1DAnnual(t,NcFile1Doutput):
    
    lon_name = 'lon'
    lat_name = 'lat'
    fy_1D= t.mean(dim=(lat_name, lon_name), skipna=True)
    
    print("CMEMS SST Dimension:",fy_1D)

    fy_1D.to_dataframe()
    
    print ('saving to ', NcFile1Doutput)
    fy_1D.to_netcdf(path=NcFile1Doutput)
    print ('finished saving')
    
    return fy_1D

##############################################################################################
##############################################################################################
############################  Winter Mean JFMA ###############################################
############################  Summer Mean JASO ###############################################
############################  GRAPH ###############################################
############################  GRAPH ###############################################
##############################################################################################
##############################################################################################


############################  NC MD MEAN DIM to CSV ###############################################

def acGenerate1DAnnualcsv(fy_1D,NcFile1DoutputCSV):
    nc = netCDF4.Dataset(fy_1D, mode='r')

    nc.variables.keys()

    time_var = nc.variables['time']
    dtime = netCDF4.num2date(time_var[:],time_var.units)
    temp = nc.variables['thetao'][:]
    
    temp_ts = pd.Series(temp, index=dtime) 
    
    temp_ts.to_csv(NcFile1DoutputCSV,index=True)

    file2 = pd.read_csv(NcFile1DoutputCSV)
    headerList = ['DATE', 'TEMPERATURE']
    file2.to_csv(NcFile1DoutputCSV, header=headerList, index=False)
    
    return temp_ts

#################################################################################
############################# ADD LIBRARY##########################################
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
############################# ADD LIBRARY########################################
#################################################################################

############################  CSV MEAN DIM ###############################################
def acGenerateDailyTimeSeries(temp_ts):
    
    file2 = pd.read_csv(temp_ts,index_col='DATE', parse_dates=['DATE'])
    
    plt.title('Temperature at Sea Surface from 1987 to 2019 in the Adriatic Sea', size=20)

    plt.plot(file2, color='r')
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    
############################  CSV MEAN DIM###############################################
    
def acGenerateDailyTimeSeriesSTD(temp_ts):
    
    file2 = pd.read_csv(temp_ts,index_col='DATE', parse_dates=True)
    fy_dt = file2.groupby(pd.Grouper(freq='M')).mean()
    
    
    daily_sdT = fy_dt.rolling(window = 12).std()
    
    #plt.plot(daily_sdT, color='k',label='STD', linewidth=3)
    plt.title('STD Temperature at Sea Surface from 1987 to 2019 in the Adriatic Sea', size=20)

    plt.plot(daily_sdT, color='r')
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    
    return daily_sdT
    
############################  CSV MEAN DIM ###############################################

def acGenerateDailyTimeSeriesPLOT(temp_ts):   
    
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