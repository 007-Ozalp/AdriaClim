import os
import numpy as np
from scipy import stats
import xarray as xr
import pandas as pd
import netCDF4
from shapely import geometry as g

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



def _get3DMaskOnPolygon(lon, lat, map3D, polygon):
    # polygon is given in form of [xcoordinates, ycoordinates]
    # assuming that map3D is (z, y, x)
    lonflatten = lon.flatten()
    latflatten = lat.flatten()
    pts = np.array([lonflatten, latflatten]).transpose()
  
    ply = g.Polygon(polygon)
    ncnt = np.vectorize(lambda p: ply.contains(g.Point(p)), signature='(n)->()')
    maskFlatten = ncnt(pts)
    mask = maskFlatten.reshape(lon.shape)
  
    nz = map3D.shape[0]
    mask_ = np.expand_dims(mask, 0)
    mask3D = mask_[np.zeros([nz]).astype(int), :, :]
  
    return mask3D

        
def acClipDataOnRegion(dataInputNcSpec, areaPerimeter, dataOutputNcFpath):
       
    """ CLIP INPUT OVER THE AREA OF INTEREST.
     Input File
     dataInputNcSpec: instance of acNcFileSpec describing the input nc file.
     areaPerimeter: pandas dataset delimiting the area being analysed. In the dataset, the 1st column is longitude, 
     the 2nd column is latitude 
     dataOutputNcFpath: path of the output nc file.
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
    
    hasZCoord = dataInputNcSpec.zVarName != ""

    #ensuring that the dimensions are in the correct order
    if hasZCoord:
      t = t.transpose(dataInputNcSpec.tVarName, dataInputNcSpec.zVarName, dataInputNcSpec.yVarName, dataInputNcSpec.xVarName)
    else:
      t = t.transpose(dataInputNcSpec.tVarName, dataInputNcSpec.yVarName, dataInputNcSpec.xVarName)

    print ('preselecting the mininumn containing rectangle, saving to ', dataOutputNcFpath)
    if os.path.isfile(dataOutputNcFpath):
      os.remove(dataOutputNcFpath)
    t.to_netcdf(path=dataOutputNcFpath)
    t.close()

    print ('clipping over the polygon and storing frame by frame (may take a while ...)')
    ds = netCDF4.Dataset(dataOutputNcFpath, "r+")
    lon = ds.variables[dataInputNcSpec.xVarName][:]
    lat = ds.variables[dataInputNcSpec.yVarName][:]
    lonmtx, latmtx = np.meshgrid(lon, lat)
    nframe = ds.variables[dataInputNcSpec.tVarName].shape[0]
    varnc = ds.variables[dataInputNcSpec.varName]
    mask3d = None
    for ifrm in range(nframe):
      if ifrm % 100 == 0:
        percDone = ifrm/nframe*100
        print(f"  done {percDone:2.0f} %", end="\r")
      vls = varnc[ifrm, :]
      vls3d = vls if hasZCoord else np.expand_dims(vls, 0)
      if mask3d is None:
        mask3d = _get3DMaskOnPolygon(lonmtx, latmtx, vls3d, areaPerimeter.values)
      clp3d = vls3d.copy()
      clp3d[~mask3d] = np.nan
      clp = clp3d if hasZCoord else clp3d[0,:]
      varnc[ifrm, :] = clp
    ds.close()
    print("\n  done!")

    

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
    if inputNcSpec.zVarName == "":
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
    inputDs = xr.open_dataset(annualMapsNcSpec.ncFileName)
  
    def _compSenSlope(vals):
      alpha = .95
      medslope, _, _, _ = stats.mstats.theilslopes(vals, alpha=alpha)
      return medslope
  
    slp = xr.apply_ufunc(_compSenSlope, inputDs, input_core_dims=[["year"]], dask="allowed", vectorize=True)
    slp.to_netcdf(outputNcFile)
    return slp

