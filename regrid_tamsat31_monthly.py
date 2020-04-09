'''
Regrid tam5 data to GPCC grid
'''
# Import libraries
import sys
import netCDF4 as nc4
import datetime as dt
import numpy as np
from netCDF4 import Dataset
from mpl_toolkits import basemap
import matplotlib.pyplot as plt
import time as tt
from netCDF4 import date2num
execfile('date_str.py')
dir_tam = '/gws/nopw/j04/tamsat/tamsat/data/rfe/v3.1/monthly/'
dir_out = '/gws/nopw/j04/ncas_climate_vol1/users/myoung02/datasets/TAMSATv3.1_monthly/'

tam_f = dir_tam+'1998/01/rfe1998_01.v3.1.nc'
nc_fid = Dataset(tam_f, 'r')
tam_lat = np.array(nc_fid.variables['lat'][:])  # extract/copy the data
tam_lon = np.array(nc_fid.variables['lon'][:])
nc_fid.close()

lonlim = [np.min(tam_lon),np.max(tam_lon)]
latlim = [np.min(tam_lat),np.max(tam_lat)]

gpcc_f = '/gws/nopw/j04/ncas_climate_vol2/users/myoung02/datasets/GPCC/monitoring_v6/monitoring_v6_10_2018_09.nc'
nc_fid = Dataset(gpcc_f, 'r')
gpcc_lat = np.array(nc_fid.variables['lat'][:])  # extract/copy the data
gpcc_lon = np.array(nc_fid.variables['lon'][:])
nc_fid.close()

gpcc_lon_id = np.where((gpcc_lon >= lonlim[0]) & (gpcc_lon <= lonlim[1]))[0]
gpcc_lat_id = np.where((gpcc_lat >= latlim[0]) & (gpcc_lat <= latlim[1]))[0]
sub_gpcc_lon = gpcc_lon[gpcc_lon_id]
sub_gpcc_lat = gpcc_lat[gpcc_lat_id]
regrid = np.meshgrid(sub_gpcc_lon,sub_gpcc_lat)
time_units = 'hours since 1900-01-01 00:00:00.0'


def do_stuff(year):
  rg_field = np.nan*np.zeros((12,len(sub_gpcc_lat),len(sub_gpcc_lon)))
  dates =[]
  for m in np.arange(0,12):
    dates.append(dt.datetime(year,m+1,1))
    # read in tamsat file
    tam_f = dir_tam+str(year)+'/'+mon_string(m+1)+'/rfe'+str(year)+'_'+mon_string(m+1)+'.v3.1.nc'
    nc_fid = Dataset(tam_f,'r')
    field = np.array(nc_fid.variables['rfe'][:]).squeeze()
    field[field<0]=np.nan
    field_units = str(nc_fid.variables['rfe'].units)
    nc_fid.close()
    rg_field[m,:,:] = basemap.interp(np.flip(field,axis=0),tam_lon,np.flip(tam_lat),regrid[0],regrid[1],order=1)

  # save new field in netcdf
  # save all data as netcdf
  nc_outfile = dir_out+'TAMSATv3.1_monthly_1d_gpcc_'+str(year)+'.nc'
  dataset = Dataset(nc_outfile,'w',format='NETCDF3_CLASSIC')
  lat = dataset.createDimension('lat', len(sub_gpcc_lat)) # create lat (dims depend on region)
  lon = dataset.createDimension('lon', len(sub_gpcc_lon)) # create lon
  time = dataset.createDimension('time', 12) # create time
  # create variables
  var_out = dataset.createVariable('rfe', 'd',('time','lat','lon'))
  latitudes = dataset.createVariable('latitude','f',('lat',))
  longitudes = dataset.createVariable('longitude','f',('lon',))
  times = dataset.createVariable('time', np.float64, ('time',))

  # Global Attributes (will need modified accordingly)
  dataset.description = 'TAMSATv3.1 monthly total rainfall regridded to GPCC 1.0d grid'
  dataset.history = 'Created ' + tt.ctime(tt.time())
  dataset.source = 'Subset by M. Young'
  # Variable Attributes
  latitudes.units = 'degrees_north'
  longitudes.units = 'degrees_east'
  var_out.units = field_units
  times.units = time_units
  times.calendar = 'gregorian'

  # Fill variables with data
  latitudes[:] = sub_gpcc_lat
  longitudes[:] = sub_gpcc_lon
  var_out[:] = rg_field
  times[:] = date2num(dates,units=time_units,calendar=times.calendar)
  dataset.close()
  return []

if __name__ == "__main__":
  output = do_stuff(int(sys.argv[1]))
