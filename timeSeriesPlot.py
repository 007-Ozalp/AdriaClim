#TODO: this function should be moved to adriaClimIndUtilsPlots
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import adriaClimIndUtilsStats as acStats


fontSizeLegend = 15
fontSizeTickLabels = 13
fontSizeAxisLabel = 16


def acPlotSSTTimeSeries(dailySSTCsv):
  dateColId = 0
  sstColId = 1

 #loading and grouping by year/month
  ds = pd.read_csv(dailySSTCsv)
  ds.iloc[:,dateColId] = pd.to_datetime(ds.iloc[:,dateColId])
  dtCol = ds.iloc[:,dateColId]
  sstCol = ds.iloc[:,sstColId]
  mts = ds.groupby([(dtCol.dt.year), (dtCol.dt.month)]).mean()
  
 #subtracting the mean seasonal cycle to get the monthly anomaly
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
  medslope, medintercept, lo_slope, up_slope, pvalue = acStats.computeAnnualTheilSenFitFromDailyFile(dailySSTCsv)
  confInt = (up_slope - lo_slope)/2

  f = plt.figure(figsize=[15, 7])
  plt.plot(dtmAnmlMon, anmlMon, color="teal", linewidth=1, label="Monthly anomaly")
  plt.plot(dtmAnmlYr, anmlYr, marker='o', markerfacecolor="firebrick", markeredgecolor='k', linewidth=0, label="Yearly anomaly")
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


  
  

