#!/usr/bin/env python
import cdsapi
import datetime
import numpy as np
import sys
execfile("date_str.py")
c = cdsapi.Client()

outputdir = "/gws/nopw/j04/ncas_climate_vol1/users/myoung02/datasets/era5/hourly/"

time_ls = ["00:00","01:00","02:00","03:00", "04:00", "05:00","06:00", "07:00", "08:00","09:00", "10:00", "11:00","12:00", "13:00", "14:00","15:00", "16:00", "17:00","18:00", "19:00", "20:00","21:00", "22:00", "23:00"]

def do_stuff(year,month,var_name):
  month_str = mon_string(month)
  nday = dayinmo(year,month)
  day_ls = []
  for d in np.arange(1,nday+1,1):
    day_ls.append(day_string(d))

  output_fname = outputdir+"era5_hrly_"+var_name+"_"+month_str+"_"+str(year)+".nc"
  c.retrieve("reanalysis-era5-single-levels",
  {"product_type": "reanalysis",
   "format":"netcdf",
   "variable":var_name,
   "area":"40/-30/-40/70", # North, West, South, East. Default: global
   "year":year,
   "month":month_str,
   "day":day_ls,
   "time":time_ls},
   output_fname)

  return []
if __name__ == "__main__":
   output = do_stuff(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
