import numpy as np
from scipy import stats

import xarray
import pandas as pd
  


def computeAnnualTheilSenFitFromDailyFile(dailyCsvFile):
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
  dsy = ds.groupby(ds.DATE.dt.year)[valColName].agg("mean")

  vals = dsy.values
  alpha = .95
  medslope, medintercept, lo_slope, up_slope = stats.mstats.theilslopes(vals, alpha=alpha)
  
  tii = np.arange(len(vals))
  kendallTau, pvalue = stats.kendalltau(tii, vals)

  return medslope, medintercept, lo_slope, up_slope, pvalue



def computeSenSlopeMap(annualMapsNcFile, outputNcFile):
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
  


