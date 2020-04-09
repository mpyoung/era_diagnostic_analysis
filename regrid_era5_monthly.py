'''
Regrid ERA5 data to GPCC grid
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


dir_era = '/gws/nopw/j04/ncas_climate_vol1/users/myoung02/datasets/era5/'
dir_out = dir_era+'regridded/'

era_f = dir_era+'era5_monthly_u_component_of_wind_850_1991.nc'
nc_fid = Dataset(era_f, 'r')
era_lat = np.array(nc_fid.variables['latitude'][:])  # extract/copy the data
era_lon = np.array(nc_fid.variables['longitude'][:])
era_u = np.array(nc_fid.variables['u'][:])
nc_fid.close()

lonlim = [np.min(era_lon),np.max(era_lon)]
latlim = [np.min(era_lat),np.max(era_lat)]

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

var_ls = np.array(['relative_humidity','vertical_velocity','u_component_of_wind','v_component_of_wind'])
var_nc = np.array(['r','w','u','v'])

def do_stuff(year,var_name,p_level):
  var_id = np.where(var_ls == var_name)[0][0]
  # read in era interim file
  era_f = dir_era+'era5_monthly_'+var_name+'_'+p_level+'_'+str(year)+'.nc'
  nc_fid = Dataset(era_f,'r')
  field = np.array(nc_fid.variables[var_nc[var_id]][:])
  field_units = str(nc_fid.variables[var_nc[var_id]].units)
  field_time = np.array(nc_fid.variables['time'][:])
  field_time_units = np.array(nc_fid.variables['time'].units)
  nc_fid.close()
  f_shape = field.shape
  if len(f_shape) == 3:
    rg_field = np.nan*np.zeros((len(field[:,0,0]),len(sub_gpcc_lat),len(sub_gpcc_lon)))
    for t in np.arange(0,len(field[:,0,0])):
      rg_field[t,:,:] = basemap.interp(np.flip(field[t,:,:],axis=0),era_lon,np.flip(era_lat),regrid[0],regrid[1],order=1)
  elif len(f_shape) == 4:
    rg_field = np.nan*np.zeros((len(field[:,0,0,0]),len(sub_gpcc_lat),len(sub_gpcc_lon)))
    for t in np.arange(0,len(field[:,0,0,0])):
      rg_field[t,:,:] = basemap.interp(np.flip(field[t,0,:,:],axis=0),era_lon,np.flip(era_lat),regrid[0],regrid[1],order=1)
  # save new field in netcdf
  # save all data as netcdf
  nc_outfile = dir_out+'era5_monthly_1d_gpcc_'+var_name+'_'+p_level+'_'+str(year)+'.nc'
  dataset = Dataset(nc_outfile,'w',format='NETCDF3_CLASSIC')
  lat = dataset.createDimension('lat', len(sub_gpcc_lat)) # create lat (dims depend on region)
  lon = dataset.createDimension('lon', len(sub_gpcc_lon)) # create lon
  time = dataset.createDimension('time', len(field[:,0,0])) # create time
  # create variables
  var_out = dataset.createVariable(var_nc[var_id], 'd',('time','lat','lon'))
  latitudes = dataset.createVariable('latitude','f',('lat',))
  longitudes = dataset.createVariable('longitude','f',('lon',))
  times = dataset.createVariable('time', np.float64, ('time',))

  # Global Attributes (will need modified accordingly)
  dataset.description = 'ERA5 monthly mean regridded to GPCC 1.0d grid'
  dataset.history = 'Created ' + tt.ctime(tt.time())
  dataset.source = 'Subset by M. Young'
  # Variable Attributes
  latitudes.units = 'degrees_north'
  longitudes.units = 'degrees_east'
  var_out.units = field_units
  times.units = field_time_units
  times.calendar = 'gregorian'

  # Fill variables with data
  latitudes[:] = sub_gpcc_lat
  longitudes[:] = sub_gpcc_lon
  var_out[:] = rg_field
  times[:] = field_time
  dataset.close()
  return []

if __name__ == "__main__":
  output = do_stuff(int(sys.argv[1]),sys.argv[2],sys.argv[3])
