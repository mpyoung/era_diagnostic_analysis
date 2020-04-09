#!/usr/bin/env python
import cdsapi
import datetime
import numpy as np
import sys

execfile("date_str.py")
c = cdsapi.Client()

outputdir = "/gws/nopw/j04/ncas_climate_vol1/users/myoung02/datasets/era5/"

# # List the forecast dates # date-month
# months = np.arange(1,13,1)#[3]#[3,6,9,12]#np.arange(3,10+1,1)
# past_years = np.array(np.arange(1983,2018+1,1),dtype="S")
#
# month = 12
# past_years = []
# for y in np.arange(1983,2018+1,1):
#   past_years.append(str(y))
#
# #
# # Do NOT ask for different start months in the same request when retrieving data from the "daily" datasets.
# # It is much more efficient to request one single month (e.g. November) for all the hindcast years (1993 to 2016) than requesting all 12 months from a single year
# #for month in months:
# var_name = "relative_humidity"
# p_levels = ["250",""]
# month_str = mon_string(month)

def do_stuff(year,var_name,p_level):

  output_fname = outputdir+"era5_monthly_"+var_name+'_'+p_level+'_'+str(year)+".nc"
  c.retrieve("reanalysis-era5-pressure-levels-monthly-means",
    {"product_type":"monthly_averaged_reanalysis",
     "variable":var_name,
     "pressure_level":p_level,
     "year":year,
     "month":["01","02","03","04","05","06","07","08","09","10","11","12"],
     "time":"00:00",
     "area":"40/-30/-40/70", # North, West, South, East. Default: global
     "format":"netcdf"},
     output_fname)
  return []
if __name__ == "__main__":
   output = do_stuff(int(sys.argv[1]),sys.argv[2],sys.argv[3])
