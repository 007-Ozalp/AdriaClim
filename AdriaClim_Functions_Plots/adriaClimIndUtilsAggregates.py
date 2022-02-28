import xarray as xr
import pandas as pd
import csv
import netCDF4


class acNcFileSpec:
    
    
    def __init__(self, ncFileName="",VarName="", xVarName="", yVarName="", zVarName="", tVarName=""):
        
            
        self.ncFileName = ncFileName
        self.VarName = VarName
        self.xVarName = xVarName
        self.yVarName = yVarName
        self.zVarName = zVarName
        self.tVarName = tVarName

    def printSpec(self):
        
            
        print(f"  ncFileName: {self.ncFileName}")
        print(f"  VarName: {self.VarName}")
        print(f"  xVarName: {self.xVarName}")
        print(f"  yVarName: {self.yVarName}")
        print(f"  zVarName: {self.zVarName}")
        print(f"  tVarName: {self.tVarName}")
        
def acClipDataOnRegion(dataInputNc, areaPerimeter, dataOutputNcFpath):
       
        """ CLIP INPUT OVER INTERESTED AREA.
         Input File
         dataInputNc: instance of acNcFileSpec describing the input nc file.
         areaPerimeter: pandas dataset delimiting the area being analysed. In the dataset, the 1st column is longitude, 
         the 2nd column is latitude 
         dataOutputNcFpath: path of the output nc file.
         
        """
    
        print("CMEMS SST Dimension:",dataInputNc)
        print("Clipped Area Dimensions:",areaPerimeter)

        iLonCol = 0
        iLatCol = 1
        areaPerLon = areaPerimeter.iloc[:,iLonCol]
        areaPerLat = areaPerimeter.iloc[:,iLatCol]
        lat_max = areaPerLat.max()
        lat_min = areaPerLat.min()
        lon_max = areaPerLon.max()
        lon_min = areaPerLon.min()
        

        inputNc = xr.open_dataset(dataInputNc.ncFileName)
        nclon = inputNc[dataInputNc.xVarName]
        nclat = inputNc[dataInputNc.yVarName]
        t = inputNc.sel({dataInputNc.yVarName:slice(lat_min,lat_max), dataInputNc.xVarName:slice(lon_min,lon_max)})
        
        print("Reseized Area:",t)

        print ('saving to ', dataOutputNcFpath)
        t.to_netcdf(path=dataOutputNcFpath)
        print ('finished saving')

        return t
    
def acGenerate2DAnnualMeanMaps(t,annualMapsNcFile): 
    
    """ Annual Mean map on previously clipped data within 33 years
    """
    def AM(month):
    
        return (month >= 1) & (month <= 12)
    
    lon_name   = t.lon[:]
    lat_name   = t.lat[:]
    
    am1 = t.sel(time=AM(t['time.month']))
    am2 = am1.groupby('time.year').mean('time')
    
    print ('ANNUAL MEAN for 33 years:', am2)
    
    print ('saving to ', annualMapsNcFile)
    am2.thetao.to_netcdf(path=annualMapsNcFile)
    print ('finished saving')
    print("Annual Mean minimum T:", am2.thetao.min())
    print("Annual Mean maximum T:", am2.thetao.max())

def acGenerate2DSeasonalWinter(t,NCoutputWINTER):
    
    """ Winter Period time selection for the creation of WINTER PERIOD NetCDF file, over previously clipped data
    """    
    
    def WINTER(month):
    
        return (month >= 1) & (month <= 4)
    
    lon_name   = t.lon[:]
    lat_name   = t.lat[:]
    
    seasonal_data_winter = t.sel(time=WINTER(t['time.month']))
    seasonal_data_winter1 = seasonal_data_winter.groupby('time.year').mean()

    print("Reseized Area:",seasonal_data_winter1)
    print("WINTER SEASON MINIMUM TEMPERATURE AT SEA SURFACE:", seasonal_data_winter1.thetao.min())
    print("WINTER SEASON MAXIMUM TEMPERATURE AT SEA SURFACE:", seasonal_data_winter1.thetao.max())

    print ('saving to ', NCoutputWINTER)
    seasonal_data_winter1.to_netcdf(path=NCoutputWINTER)
    print ('finished saving')
    
    return seasonal_data_winter
    
def acGenerate2DSeasonalSummer(t,NCoutputSUMMER):
    
    """ Summer Period time selection for the creation of SUMMER PERIOD NetCDF file, over previously clipped data
    """
    
    def SUMMER(month):
        
        return (month >= 7) & (month <= 10)
    
    lon_name   = t.lon[:]
    lat_name   = t.lat[:]
    
    seasonal_data_summer = t.sel(time=SUMMER(t['time.month']))
    seasonal_data_summer1 = seasonal_data_summer.groupby('time.year').mean()
    
    print("Reseized Area:",seasonal_data_summer1)
    print("SUMMER SEASON MINIMUM TEMPERATURE AT SEA SURFACE:", seasonal_data_summer1.thetao.min())
    print("SUMMER SEASON MAXIMUM TEMPERATURE AT SEA SURFACE:", seasonal_data_summer1.thetao.max())

    print ('saving to ', NCoutputSUMMER)

    seasonal_data_summer1.to_netcdf(path=NCoutputSUMMER)
    print ('finished saving')

def acGenerate1DFixDim(t,NcFile1Doutput):
    
    """ Mean sized LAT an LON  1 dimensional NetCDF file over previously clipped data, for the next csv file creation function
    """
     
    lon_name = 'lon'
    lat_name = 'lat'
    fy_1D= t.mean(dim=(lat_name, lon_name), skipna=True)
    fy_1D.to_dataframe()
    print ('saving to ', NcFile1Doutput)
    fy_1D.to_netcdf(path=NcFile1Doutput)
    print ('finished saving') 
    print("File Dimension:",fy_1D)

    return fy_1D    


def acGenerate1DFixDimCSV(fy_1D,NcFile1DoutputCSV):
    
    """ Mean sized LAT an LON  1 dimensional CSV file over previously clipped data
    """
    
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
