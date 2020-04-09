# Import libraries
import netCDF4 as nc4
import datetime as dt
import numpy as np
from netCDF4 import Dataset

execfile('date_str.py')

# gpcc monitoring (1983 to 2019)
# inputdir = '/gws/nopw/j04/ncas_climate_vol2/users/myoung02/datasets/GPCC/monitoring_v6/'
def grab_gpcc_monitoring_region_month(lonlim,latlim,month,year):
  dir = '/gws/nopw/j04/ncas_climate_vol2/users/myoung02/datasets/GPCC/monitoring_v6/'
  #print 'loading gpcc data'
  inputfile = dir+'monitoring_v6_10_'+str(year)+'_'+mon_string(month)+'.nc'
  nc_fid = Dataset(inputfile, 'r')
  lat = np.array(nc_fid.variables['lat'][:])  # extract/copy the data
  lon = np.array(nc_fid.variables['lon'][:])
  # t_time = nc_fid.variables['time'][:]
  # # convert time to date using netCDF4 function
  # units='days since 1901-01-01 00:00:00'
  # all_dates = nc4.num2date(t_time,units)
  # # myvar = nc_fid.variables['p']
  fill_val = nc_fid.variables['p']._FillValue
  # s_var = nc_fid.variables['s']
  s_fill_val = nc_fid.variables['s']._FillValue
  #
  # gpcc_yrs = []
  # gpcc_mons = []
  # gpcc_days = []
  # for i in range(0,len(all_dates)):
  #   curr_day = all_dates[i]
  #   gpcc_yrs.append((curr_day.year))
  #   gpcc_mons.append((curr_day.month))
  #   gpcc_days.append((curr_day.day))
  # gpcc_yrs = np.array(gpcc_yrs,dtype='f')
  # gpcc_mons = np.array(gpcc_mons,dtype='f')
  # gpcc_days = np.array(gpcc_days,dtype='f')
  #
  # y_id = np.where(gpcc_yrs == year)[0]
  # m_id = np.where(gpcc_mons == month)[0]
  # time_id = np.intersect1d(y_id,m_id,assume_unique=False)
  # out_dates = all_dates[time_id]

  # grab rfe at specific latitude and longitude (inlat,inlon)
  rg_lon = np.where((lon >= lonlim[0]) & (lon <= lonlim[1]))
  rg_lat = np.where((lat >= latlim[0]) & (lat <= latlim[1]))
  match_lon = lon[rg_lon[0]]
  match_lat = lat[rg_lat[0]]
  rfe_rg = np.array(nc_fid.variables['p'][:,rg_lat[0],:][:,:,rg_lon[0]])  # shape is time, lat, lon as shown above
  #rfe_rg = rfe[:,rg_lat[0],:][:,:,rg_lon[0]]
  rfe_rg[rfe_rg == fill_val] = np.nan
  s_rg = np.array(nc_fid.variables['s'][:,rg_lat[0],:][:,:,rg_lon[0]])  # shape is time, lat, lon as shown above
  #rfe_rg = rfe[:,rg_lat[0],:][:,:,rg_lon[0]]
  s_rg[s_rg == s_fill_val] = np.nan
  nc_fid.close()

  return (rfe_rg,s_rg,match_lon,match_lat)
