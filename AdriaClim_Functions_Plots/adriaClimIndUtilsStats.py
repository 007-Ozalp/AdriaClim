import numpy as np
from scipy import stats

import xarray
import pandas as pd
  


def computeAnnualTheilSenFitFromDailyFile(dailyCsvFile):
  
  
    ds = pd.read_csv(dailyCsvFile)
  # assuming that date is the 1st column, value is the 2nd column
    dateCol = 0
    valCol = 1
  # converting object to date
    ds.iloc[:,dateCol] = pd.to_datetime(ds.iloc[:,dateCol])
    valColName = ds.iloc[:,valCol].name
    dsy = ds.groupby(ds.DATE.dt.year)[valColName].agg("mean")

    vals = dsy.values
    alpha = .95
    medslope, medintercept, lo_slope, up_slope = stats.mstats.theilslopes(vals, alpha=alpha)
  
    tii = np.arange(len(vals))
    kendallTau, pvalue = stats.kendalltau(tii, vals)
    
    print("tii:",tii)
    print("Med slope:", medslope)
    print("Med Intercept:", medintercept)
    print("lo slope:", lo_slope)
    print("Up slope:", up_slope)
    print("p_value:", pvalue)
    return medslope, medintercept, lo_slope, up_slope, pvalue



def computeSenSlopeMap(annualMapsNcFile, outputNcFile):
  
    inputDs = xarray.open_dataset(annualMapsNcFile)
    
    
        

    def _compSenSlope(vals):
        alpha = .95
        medslope, _, _, _ = stats.mstats.theilslopes(vals, alpha=alpha)
        return medslope

    slp = xarray.apply_ufunc(_compSenSlope, inputDs, input_core_dims=[["year"]], dask="allowed", vectorize=True)
    slp.to_netcdf(outputNcFile)
    print("Output:", slp)
    print("output  min:", slp.thetao.min())
    print("output max:", slp.thetao.max())
    return slp