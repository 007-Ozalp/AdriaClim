import os
import numpy as np
from scipy import stats
import xarray as xr
import pandas as pd

import regionmask


class acNcFileSpec:
    
    def __init__(self, ncFileName="",varName="", xVarName="", yVarName="", zVarName="", tVarName=""):
        self.ncFileName = ncFileName
        self.varName = varName
        self.xVarName = xVarName
        self.yVarName = yVarName
        self.zVarName = zVarName
        self.tVarName = tVarName

    def printSpec(self):
        print(f"  ncFileName: {self.ncFileName}")
        print(f"  varName: {self.varName}")
        print(f"  xVarName: {self.xVarName}")
        print(f"  yVarName: {self.yVarName}")
        print(f"  zVarName: {self.zVarName}")
        print(f"  tVarName: {self.tVarName}")

    def clone(self, src, **kwargs):
        self.__dict__.update(src.__dict__)
        self.__dict__.update(kwargs)


def acCloneFileSpec(src, **kwargs):
    out = acNcFileSpec()
    out.clone(src, **kwargs)
    return out


        
def acClipDataOnRegion(dataInputNcSpec, areaPerimeter, dataOutputNcFpath):
       
    """ CLIP INPUT OVER INTERESTED AREA.
     Input File
     dataInputNcSpec: instance of acNcFileSpec describing the input nc file.
     areaPerimeter: pandas dataset delimiting the area being analysed. In the dataset, the 1st column is longitude, 
     the 2nd column is latitude 
     dataOutputNcFpath: path of the output nc file.
     TODO: the region must be clipped on the polygon, not on the rectangle
    """
    
    print("CMEMS SST Dimension:",dataInputNcSpec)
    print("Clipped Area Dimensions:",areaPerimeter)

    iLonCol = 0
    iLatCol = 1
    areaPerLon = areaPerimeter.iloc[:,iLonCol]
    areaPerLat = areaPerimeter.iloc[:,iLatCol]
    lat_max = areaPerLat.max()
    lat_min = areaPerLat.min()
    lon_max = areaPerLon.max()
    lon_min = areaPerLon.min()
    

    inputNc = xr.open_dataset(dataInputNcSpec.ncFileName)
    nclon = inputNc[dataInputNcSpec.xVarName]
    nclat = inputNc[dataInputNcSpec.yVarName]
    t = inputNc.sel({dataInputNcSpec.yVarName:slice(lat_min,lat_max), dataInputNcSpec.xVarName:slice(lon_min,lon_max)})
    
    print("Reseized Area:",t)

    print ('saving to ', dataOutputNcFpath)
    if os.path.isfile(dataOutputNcFpath):
      os.remove(dataOutputNcFpath)
    t.to_netcdf(path=dataOutputNcFpath)
    print ('finished saving')

   #TODO: clip to the polygon
   #rgn = regionmask.Regions([np.array(areaPerimeter)])
   #msk = rgn.mask(t0[dataInputNcSpec.varName], method="shapely")
   #t = xr.where(msk, t0, np.nan)

    return t
    

def __acGenerate2DAnnualMeanMaps(inputNcSpec, outputFileName, timeSelector):
    inDt = xr.open_dataset(inputNcSpec.ncFileName)
    xx = inDt[inputNcSpec.xVarName]
    yy = inDt[inputNcSpec.yVarName]
    tm = inDt[inputNcSpec.tVarName]

    _sel = inDt.sel({tm.name: timeSelector(tm)})
    aggstr = f"{tm.name}.year"
    ouDt = _sel.groupby(aggstr).mean()
    if os.path.isfile(outputFileName):
      os.remove(outputFileName)
    ouDt.to_netcdf(path=outputFileName)
    


def acGenerate2DAnnualMeanMaps(inputNcSpec, annualMapsNcFile): 
    
    """ Annual Mean map on previously clipped data within 33 years
    """
    def AM(tm):
        month = tm.dt.month
        return (month >= 1) & (month <= 12)
    __acGenerate2DAnnualMeanMaps(inputNcSpec, annualMapsNcFile, AM)
    


def acGenerate2DSeasonalWinter(inputNcSpec, winterMapsNcFile):
    
    """ Winter Period time selection for the creation of WINTER PERIOD NetCDF file, over previously clipped data
    """    
    def WINTER(tm):
        month = tm.dt.month
        return (month >= 1) & (month <= 4)
    __acGenerate2DAnnualMeanMaps(inputNcSpec, winterMapsNcFile, WINTER)


    
def acGenerate2DSeasonalSummer(inputNcSpec, summerMapsNcFile):
    
    """ Summer Period time selection for the creation of SUMMER PERIOD NetCDF file, over previously clipped data
    """
    def SUMMER(tm):
        month = tm.dt.month
        return (month >= 7) & (month <= 10)
    __acGenerate2DAnnualMeanMaps(inputNcSpec, summerMapsNcFile, SUMMER)



def acGenerateMeanTimeSeries(inputNcSpec, outCsvFile):
    
    """ Mean sized LAT an LON  1 dimensional NetCDF file over previously clipped data, for the next csv file creation function
    """
    inDs = xr.open_dataset(inputNcSpec.ncFileName) 
    if inputNcSpec.zVarName is "":
      fy_1D= inDs.mean(dim=(inputNcSpec.yVarName, inputNcSpec.xVarName), skipna=True)
    else:
      fy_1D= inDs.mean(dim=(inputNcSpec.yVarName, inputNcSpec.xVarName, inputNcSpec.zVarName), skipna=True)
    ouDs = fy_1D.to_dataframe()[inputNcSpec.varName]
    ouDs.to_csv(outCsvFile)
    return ouDs
  


def acComputeAnnualTheilSenFitFromDailyFile(dailyCsvFile):
    """
    computeAnnualTheilSenFitFromDailyFile: computes the Theil-Sen fit of the annual means, using an input file with daily values
    input:
      dailyCsvFile: csv file with daily values.
    output:
      medslope, medintercept, lo_slope, up_slope, pvalue. lo_slope and up_slope are computed with a 95% confidence.
    """
    
    ds = pd.read_csv(dailyCsvFile)
    # assuming that date is the 1st column, value is the 2nd column
    dateCol = 0
    valCol = 1
    # converting object to date
    ds.iloc[:,dateCol] = pd.to_datetime(ds.iloc[:,dateCol])
    valColName = ds.iloc[:,valCol].name
    dsDateCol = ds.iloc[:,dateCol]
    dsy = ds.groupby(dsDateCol.dt.year)[valColName].agg("mean")

    vals = dsy.values
    alpha = .95
    medslope, medintercept, lo_slope, up_slope = stats.mstats.theilslopes(vals, alpha=alpha)
    
    tii = np.arange(len(vals))
    kendallTau, pvalue = stats.kendalltau(tii, vals)

    return medslope, medintercept, lo_slope, up_slope, pvalue



def acComputeSenSlopeMap(annualMapsNcSpec, outputNcFile):
    """
    computeSenSlopeMap: generates a map with the sen's slope for each pixel, given the series of annual maps in file inputNcFile.
    input:
      annualMapsNcFile: input nc file with annual maps of the variable of interest. The time variable is assumed to be called "year".
    output:
      outputNcFile: file where the slope is stored.
    """
    inputDs = xarray.open_dataset(annualMapsNcFile)
  
    def _compSenSlope(vals):
      alpha = .95
      medslope, _, _, _ = stats.mstats.theilslopes(vals, alpha=alpha)
      return medslope
  
    slp = xarray.apply_ufunc(_compSenSlope, inputDs, input_core_dims=[["year"]], dask="allowed", vectorize=True)
    slp.to_netcdf(outputNcFile)
    return slp

