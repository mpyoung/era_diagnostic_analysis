'''
Script analyses monthly biases in TAMSATv3.1 relative
to the GPCC monitoring product (chosen for its longer timeseries than FDR)
fields are:
upper and lower relative_humidity
mid-level vertical velocity
vertically integrated moisture divergence

All datasets (TAMSATv3.1, ERA5 and GPCC) have been
regridded to a common 0.25 degree grid
Monthly biases are compared to monthly re-analysis fields
from ERA5.
reads in TAMSAT3.1 data at ?? (0.25?) degree GPCC grid
Computes bias with GPCC

M. Young Feb 2020
'''
from __future__ import division
import netCDF4 as nc4
import datetime as dt
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import Basemap
from mpl_toolkits import basemap
from netCDF4 import Dataset
from scipy import stats
import os
from scipy.stats import mode
from scipy.stats.stats import pearsonr
from matplotlib.pyplot import cm

# function to read the variables from netcdf file
def read_nc(file,variables):
  nc_fid = Dataset(file,'r')
  output = []
  for n in np.arange(0,len(variables)):
    output.append(np.array(nc_fid.variables[variables[n]]))
  nc_fid.close()
  return output

execfile('date_str.py')
execfile('grab_gpcc.py')

dir_out = '/home/users/myoung02/era_diagnostics2020/'
dir_era = '/gws/nopw/j04/ncas_climate_vol1/users/myoung02/datasets/era5/regridded/'
dir_gpc = '/gws/nopw/j04/ncas_climate_vol2/users/myoung02/datasets/GPCC/monitoring_v6/'
dir_tam = '/gws/nopw/j04/ncas_climate_vol1/users/myoung02/datasets/TAMSATv3.1_monthly/'

# spatial parameters
# lonlim = [-20,55]
# latlim = [-36,40]
region='Africa'
years = np.arange(1983,2019+1,1)
months = np.arange(1,12+1,1)
month_names = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

# subset everything to tamsat 1 degree grid
f_tam = dir_tam+'TAMSATv3.1_monthly_1d_gpcc_1999.nc'
tam_grid = read_nc(f_tam,['longitude','latitude'])
lonlim = [tam_grid[0][0],tam_grid[0][-1]]
latlim = [tam_grid[1][-1],tam_grid[1][0]]

f_era = dir_era+'era5_monthly_1d_gpcc_vertical_velocity_500_1999.nc'
era_grid = read_nc(f_era,['longitude','latitude'])
era_lon_id = np.where((era_grid[0] >= lonlim[0]) & (era_grid[0] <= lonlim[1]))[0]
era_lat_id = np.where((era_grid[1] >= latlim[0]) & (era_grid[1] <= latlim[1]))[0]


# create empty grid
gpcc_f = dir_gpc+'monitoring_v6_10_2018_09.nc'
nc_fid = Dataset(gpcc_f, 'r')
gpcc_lat = np.array(nc_fid.variables['lat'][:])  # extract/copy the data
gpcc_lon = np.array(nc_fid.variables['lon'][:])
nc_fid.close()
gpcc_lon_id = np.where((gpcc_lon >= lonlim[0]) & (gpcc_lon <= lonlim[1]))[0]
gpcc_lat_id = np.where((gpcc_lat >= latlim[0]) & (gpcc_lat <= latlim[1]))[0]
sub_gpcc_lon = gpcc_lon[gpcc_lon_id]
sub_gpcc_lat = gpcc_lat[gpcc_lat_id]



var_ls = np.array(['relative_humidity','vertical_velocity','u_component_of_wind','v_component_of_wind','vertically_integrated_moisture_divergence'])
var_nc = np.array(['r','w','u','v','vimd'])
p_lev= ['250','850','500']

all_rfe = np.nan*np.zeros((len(years),len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))
all_gpcc = np.nan*np.zeros((2,len(years),len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))
era_w = np.nan*np.zeros((len(years),len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))
era_mf = np.nan*np.zeros((len(years),len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))
era_u = np.nan*np.zeros((2,len(years),len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))
era_v = np.nan*np.zeros((2,len(years),len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))
era_rh = np.nan*np.zeros((2,len(years),len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))

# open gpcc, tamsat and era5 1 degree data
for y in np.arange(0,len(years)):
  print years[y]
  f_tam = dir_tam+'TAMSATv3.1_monthly_1d_gpcc_'+str(years[y])+'.nc'
  all_rfe[y,:,:,:] = read_nc(f_tam,['rfe'])[0]

  f_era_w = dir_era+'era5_monthly_1d_gpcc_vertical_velocity_500_'+str(years[y])+'.nc'
  era_w[y,:,:,:] = read_nc(f_era_w,['w'])[0][:,era_lat_id,:][:,:,era_lon_id]

  f_era_mf = dir_era+'era5_monthly_1d_gpcc_vertically_integrated_moisture_divergence_'+str(years[y])+'.nc'
  era_mf[y,:,:,:] = read_nc(f_era_mf,['vimd'])[0][:,era_lat_id,:][:,:,era_lon_id]

  for p in [0,1]:
    f_era_u = dir_era+'era5_monthly_1d_gpcc_u_component_of_wind_'+p_lev[p]+'_'+str(years[y])+'.nc'
    era_u[p,y,:,:,:] = read_nc(f_era_u,['u'])[0][:,era_lat_id,:][:,:,era_lon_id]
    f_era_v = dir_era+'era5_monthly_1d_gpcc_v_component_of_wind_'+p_lev[p]+'_'+str(years[y])+'.nc'
    era_v[p,y,:,:,:] = read_nc(f_era_v,['v'])[0][:,era_lat_id,:][:,:,era_lon_id]
    f_era_rh = dir_era+'era5_monthly_1d_gpcc_relative_humidity_'+p_lev[p]+'_'+str(years[y])+'.nc'
    era_rh[p,y,:,:,:] = read_nc(f_era_rh,['r'])[0][:,era_lat_id,:][:,:,era_lon_id]

  for m in np.arange(0,len(months)):
    f_gpc = dir_gpc+'monitoring_v6_10_'+str(years[y])+'_'+mon_string(months[m])+'.nc'
    if os.path.isfile(f_gpc) == True:
      tmp = []
      tmp = grab_gpcc_monitoring_region_month(lonlim,latlim,months[m],years[y])
      all_gpcc[0,y,m,:,:] = tmp[0]
      all_gpcc[1,y,m,:,:] = tmp[1]

clim_gpcc= np.nanmean(all_gpcc[0,:,:,:,:],axis=0)
rep_clim_gpcc = np.repeat(clim_gpcc[np.newaxis,:,:,:],len(years),axis=0)
clim_rfe = np.nanmean(all_rfe,axis=0)
rep_clim_rfe = np.repeat(clim_rfe[np.newaxis,:,:,:],len(years),axis=0)
anom_rfe = all_rfe - rep_clim_rfe
anom_gpcc = all_gpcc[0,:,:,:,:] - rep_clim_gpcc

# compute vector wind and shear
era5_fill = -32767
era_w[era_w==era5_fill] = np.nan
era_mf[era_mf==era5_fill] = np.nan
era_u[era_u==era5_fill] = np.nan
era_v[era_v==era5_fill] = np.nan
era_rh[era_rh==era5_fill] = np.nan

era_vec_wind = np.sqrt((era_u**2)+(era_v**2))
era_shear = abs(era_vec_wind[0,:,:,:,:]-era_vec_wind[1,:,:,:,:])

mean_era_vec = np.nanmean(era_vec_wind,axis=1)
mean_era_shear = np.nanmean(era_shear,axis=0)
mean_era_w = np.nanmean(era_w,axis=0)
mean_era_mf = np.nanmean(era_mf,axis=0)
mean_era_u = np.nanmean(era_u,axis=1)
mean_era_v = np.nanmean(era_v,axis=1)
mean_era_rh = np.nanmean(era_rh,axis=1)

# compute anomalies in the monthly era interim data
anom_era_vec = np.nan*np.zeros(era_vec_wind.shape)
anom_era_shear = np.nan*np.zeros(era_shear.shape)
anom_era_w = np.nan*np.zeros(era_w.shape)
anom_era_mf = np.nan*np.zeros(era_mf.shape)
anom_era_u = np.nan*np.zeros(era_u.shape)
anom_era_v = np.nan*np.zeros(era_v.shape)
anom_era_rh = np.nan*np.zeros(era_rh.shape)

for m in np.arange(0,12):
  anom_era_vec[:,:,m,:,:] = era_vec_wind[:,:,m,:,:] - np.repeat(mean_era_vec[:,np.newaxis,m,:,:],len(years),axis=1)
  anom_era_shear[:,m,:,:] = era_shear[:,m,:,:] - np.repeat(mean_era_shear[np.newaxis,m,:,:],len(years),axis=0)
  anom_era_w[:,m,:,:] = era_w[:,m,:,:] - np.repeat(mean_era_w[np.newaxis,m,:,:],len(years),axis=0)
  anom_era_mf[:,m,:,:] = era_mf[:,m,:,:] - np.repeat(mean_era_mf[np.newaxis,m,:,:],len(years),axis=0)
  anom_era_u[:,:,m,:,:] = era_u[:,:,m,:,:] - np.repeat(mean_era_u[:,np.newaxis,m,:,:],len(years),axis=1)
  anom_era_v[:,:,m,:,:] = era_v[:,:,m,:,:] - np.repeat(mean_era_v[:,np.newaxis,m,:,:],len(years),axis=1)
  anom_era_rh[:,:,m,:,:] = era_rh[:,:,m,:,:] - np.repeat(mean_era_rh[:,np.newaxis,m,:,:],len(years),axis=1)

# gpcc stations
st_tot_yr = np.nansum(all_gpcc[1,:,:,:,:],axis=(1,2,3))
# compute bias
all_bias = all_rfe-all_gpcc[0,:,:,:,:]
mon_bias = np.nanmean(all_bias,axis=0)
mean_bias = np.nanmean(all_bias,axis=(0,1))
# mask bias over regions where climatological rainfall < 50mm
mask_gpcc= np.copy(clim_gpcc)
mask_gpcc[mask_gpcc<50] = np.nan
mask_gpcc[mask_gpcc>=50] = 1
rep_mask_gpcc = np.repeat(mask_gpcc[np.newaxis,:,:,:],len(years),axis=0)


'''
1. Plot overall TAMSATv3.1 bias w.r.t GPCC
'''
cmin = -50
cmax = 50
cspc = 10
clevs = np.arange(cmin,cmax+cspc,cspc)
norm = BoundaryNorm(boundaries=clevs, ncolors=256)
lw =1

# Monthly mean bias
fname_plot = dir_out+'TAMSATv3.1_GPCC_bias_'+str(years[0])+'_'+str(years[-1])
fig = plt.figure(figsize=(3,3))
mymap = Basemap(projection='cyl',resolution='l',\
      llcrnrlat=latlim[0],urcrnrlat=latlim[1],\
      llcrnrlon=lonlim[0],urcrnrlon=lonlim[1])
mymap.drawcoastlines(linewidth=lw)
x, y = mymap(*np.meshgrid(sub_gpcc_lon,sub_gpcc_lat))
tmp_bias = []
tmp_bias = np.copy(mean_bias)
tmp_bias[tmp_bias == 0] = np.nan
uncal1 = mymap.pcolormesh(x,y,tmp_bias,cmap='RdYlBu',vmin=cmin,vmax=cmax,norm=norm)
# plt.colorbar(label='Bias (mm)',orientation='horizontal')
  #plt.title(str(m+1))
fig.subplots_adjust(bottom=0.15)
cbar_pos = [0.1, 0.005, 0.8, 0.05] #[left, bottom, width, height]
cbar_ax = fig.add_axes(cbar_pos)
cbar = fig.colorbar(uncal1,cax=cbar_ax,label='Bias (mm)',orientation='horizontal')
plt.tight_layout()
plt.savefig(fname_plot+'.png',bbox_inches='tight',dpi=300)
plt.close()

# Mean bias for each month of year
cmin = -100
cmax = 100
cspc = 25
clevs = np.arange(cmin,cmax+cspc,cspc)
norm = BoundaryNorm(boundaries=clevs, ncolors=256)
lw =1
fname_plot = dir_out+'TAMSATv3.1_GPCC_monthly_bias_'+str(years[0])+'_'+str(years[-1])
fig = plt.figure(figsize=(10,8))
for m in np.arange(0,12):
  plt.subplot(3,4,m+1)
  mymap = Basemap(projection='cyl',resolution='l',\
        llcrnrlat=latlim[0],urcrnrlat=latlim[1],\
        llcrnrlon=lonlim[0],urcrnrlon=lonlim[1])
  mymap.drawcoastlines(linewidth=lw)
  x, y = mymap(*np.meshgrid(sub_gpcc_lon,sub_gpcc_lat))
  tmp_bias = []
  tmp_bias = np.copy(mon_bias[m,:,:])
  tmp_bias[tmp_bias == 0] = np.nan
  uncal1 = mymap.pcolormesh(x,y,tmp_bias,cmap='RdYlBu',vmin=cmin,vmax=cmax,norm=norm)
  # plt.colorbar()
  #plt.title(str(m+1))
  plt.text(-15,-30,month_names[m],fontsize=14,fontweight='bold')
fig.subplots_adjust(bottom=0.15)
cbar_pos = [0.325, 0.025, 0.35, 0.01] #[left, bottom, width, height]
cbar_ax = fig.add_axes(cbar_pos)
cbar = fig.colorbar(uncal1,cax=cbar_ax,label='Bias (mm)',orientation='horizontal')
plt.tight_layout(pad=2.5,w_pad=0.02,h_pad=0.1)
plt.savefig(fname_plot+'.png',bbox_inches='tight',dpi=300)
plt.close()


'''
Plot case study bias - DJF 2015/2016 and OND 2019
'''
y_id = np.array((np.where(years == 2015)[0],np.where(years == 2016)[0],np.where(years == 2016)[0]))
m_id = np.array((np.where(months == 12)[0],np.where(months == 1)[0],np.where(months == 2)[0]))
# Monthly mean bias
cmin = -100
cmax = 100
cspc = 25
clevs = np.arange(cmin,cmax+cspc,cspc)
norm = BoundaryNorm(boundaries=clevs, ncolors=256)
lw =1
fname_plot = dir_out+'TAMSATv3.1_GPCC_DJF_2015_2016_bias_'+str(years[0])+'_'+str(years[-1])
fig = plt.figure(figsize=(8,8))
for m in np.arange(0,len(m_id)):
  plt.subplot(3,len(m_id),m+1)
  mymap = Basemap(projection='cyl',resolution='l',\
        llcrnrlat=latlim[0],urcrnrlat=latlim[1],\
        llcrnrlon=lonlim[0],urcrnrlon=lonlim[1])
  mymap.drawcoastlines(linewidth=lw)
  x, y = mymap(*np.meshgrid(sub_gpcc_lon,sub_gpcc_lat))
  tmp_bias = []
  tmp_bias = np.copy(anom_gpcc[y_id[m],m_id[m],:,:]).squeeze()
  tmp_bias[tmp_bias == 0] = np.nan
  p_anom = mymap.pcolormesh(x,y,tmp_bias,cmap='RdYlBu',vmin=cmin,vmax=cmax,norm=norm)
  if m == 0:
    plt.ylabel('GPCC anomaly',fontsize=12,fontweight='bold')
  plt.title(month_names[m_id[m][0]]+' '+str(years[y_id[m][0]]))

  plt.subplot(3,len(m_id),m+4)
  mymap = Basemap(projection='cyl',resolution='l',\
        llcrnrlat=latlim[0],urcrnrlat=latlim[1],\
        llcrnrlon=lonlim[0],urcrnrlon=lonlim[1])
  mymap.drawcoastlines(linewidth=lw)
  x, y = mymap(*np.meshgrid(sub_gpcc_lon,sub_gpcc_lat))
  tmp_bias = []
  tmp_bias = np.copy(anom_rfe[y_id[m],m_id[m],:,:]).squeeze()
  tmp_bias[tmp_bias == 0] = np.nan
  p_anom1 = mymap.pcolormesh(x,y,tmp_bias,cmap='RdYlBu',vmin=cmin,vmax=cmax,norm=norm)
  if m == 0:
    plt.ylabel('TAMSATv3.1 anomaly',fontsize=12,fontweight='bold')

  plt.subplot(3,len(m_id),m+7)
  mymap = Basemap(projection='cyl',resolution='l',\
        llcrnrlat=latlim[0],urcrnrlat=latlim[1],\
        llcrnrlon=lonlim[0],urcrnrlon=lonlim[1])
  mymap.drawcoastlines(linewidth=lw)
  x, y = mymap(*np.meshgrid(sub_gpcc_lon,sub_gpcc_lat))
  tmp_bias = []
  tmp_bias = np.copy(all_bias[y_id[m],m_id[m],:,:]).squeeze()
  tmp_bias[tmp_bias == 0] = np.nan
  p_bias = mymap.pcolormesh(x,y,tmp_bias,cmap='RdYlBu',vmin=cmin,vmax=cmax,norm=norm)
  if m == 0:
    plt.ylabel('TAMSATv3.1 bias',fontsize=12,fontweight='bold')
  # plt.colorbar()
  #plt.title(str(m+1))
  # plt.text(-17,-33,month_names[m_id[m][0]]+' '+str(years[y_id[m][0]]),fontsize=14,fontweight='bold')
fig.subplots_adjust(right=0.9)
cbar_pos = [0.96, 0.5, 0.017, 0.3] #[left, bottom, width, height]
cbar_ax = fig.add_axes(cbar_pos)
cbar = fig.colorbar(p_anom,cax=cbar_ax,label='Anomaly (mm)',orientation='vertical',extend='both')
cbar_pos = [0.96, 0.05, 0.017, 0.25] #[left, bottom, width, height]
cbar_ax = fig.add_axes(cbar_pos)
cbar = fig.colorbar(p_bias,cax=cbar_ax,label='Bias (mm)',orientation='vertical',extend='both')
plt.tight_layout(pad=2.5,w_pad=0.02,h_pad=0.1)
plt.savefig(fname_plot+'.png',bbox_inches='tight',dpi=300)
plt.close()


y_id = np.array((np.where(years == 2019)[0],np.where(years == 2019)[0],np.where(years == 2019)[0]))
m_id = np.array((np.where(months == 9)[0],np.where(months == 10)[0],np.where(months == 11)[0]))
# Monthly mean bias
cmin = -150
cmax = 150
cspc = 25
clevs = np.arange(cmin,cmax+cspc,cspc)
norm = BoundaryNorm(boundaries=clevs, ncolors=256)
lw =1
fname_plot = dir_out+'TAMSATv3.1_GPCC_OND_2019_bias_'+str(years[0])+'_'+str(years[-1])
fig = plt.figure(figsize=(8,8))
for m in np.arange(0,len(m_id)):

  plt.subplot(3,len(m_id),m+1)
  mymap = Basemap(projection='cyl',resolution='l',\
        llcrnrlat=latlim[0],urcrnrlat=latlim[1],\
        llcrnrlon=lonlim[0],urcrnrlon=lonlim[1])
  mymap.drawcoastlines(linewidth=lw)
  x, y = mymap(*np.meshgrid(sub_gpcc_lon,sub_gpcc_lat))
  tmp_bias = []
  tmp_bias = np.copy(anom_gpcc[y_id[m],m_id[m],:,:]).squeeze()
  tmp_bias[tmp_bias == 0] = np.nan
  p_anom = mymap.pcolormesh(x,y,tmp_bias,cmap='RdYlBu',vmin=cmin,vmax=cmax,norm=norm)
  if m == 0:
    plt.ylabel('GPCC anomaly',fontsize=12,fontweight='bold')
  plt.title(month_names[m_id[m][0]]+' '+str(years[y_id[m][0]]))

  plt.subplot(3,len(m_id),m+4)
  mymap = Basemap(projection='cyl',resolution='l',\
        llcrnrlat=latlim[0],urcrnrlat=latlim[1],\
        llcrnrlon=lonlim[0],urcrnrlon=lonlim[1])
  mymap.drawcoastlines(linewidth=lw)
  x, y = mymap(*np.meshgrid(sub_gpcc_lon,sub_gpcc_lat))
  tmp_bias = []
  tmp_bias = np.copy(anom_rfe[y_id[m],m_id[m],:,:]).squeeze()
  tmp_bias[tmp_bias == 0] = np.nan
  p_anom1 = mymap.pcolormesh(x,y,tmp_bias,cmap='RdYlBu',vmin=cmin,vmax=cmax,norm=norm)
  if m == 0:
    plt.ylabel('TAMSATv3.1 anomaly',fontsize=12,fontweight='bold')

  plt.subplot(3,len(m_id),m+7)
  mymap = Basemap(projection='cyl',resolution='l',\
        llcrnrlat=latlim[0],urcrnrlat=latlim[1],\
        llcrnrlon=lonlim[0],urcrnrlon=lonlim[1])
  mymap.drawcoastlines(linewidth=lw)
  x, y = mymap(*np.meshgrid(sub_gpcc_lon,sub_gpcc_lat))
  tmp_bias = []
  tmp_bias = np.copy(all_bias[y_id[m],m_id[m],:,:]).squeeze()
  tmp_bias[tmp_bias == 0] = np.nan
  p_bias = mymap.pcolormesh(x,y,tmp_bias,cmap='RdYlBu',vmin=cmin,vmax=cmax,norm=norm)
  if m == 0:
    plt.ylabel('TAMSATv3.1 bias',fontsize=12,fontweight='bold')
  # plt.colorbar()
  #plt.title(str(m+1))
  # plt.text(-17,-33,month_names[m_id[m][0]]+' '+str(years[y_id[m][0]]),fontsize=14,fontweight='bold')
fig.subplots_adjust(right=0.9)
cbar_pos = [0.96, 0.5, 0.017, 0.3] #[left, bottom, width, height]
cbar_ax = fig.add_axes(cbar_pos)
cbar = fig.colorbar(p_anom,cax=cbar_ax,label='Anomaly (mm)',orientation='vertical',extend='both')
cbar_pos = [0.96, 0.05, 0.017, 0.25] #[left, bottom, width, height]
cbar_ax = fig.add_axes(cbar_pos)
cbar = fig.colorbar(p_bias,cax=cbar_ax,label='Bias (mm)',orientation='vertical',extend='both')
plt.tight_layout(pad=2.5,w_pad=0.02,h_pad=0.1)
plt.savefig(fname_plot+'.png',bbox_inches='tight',dpi=300)
plt.close()


'''
Conditional bias analysis (based on large-scale drivers)
load hadisst
'''
sst_latlim = [-50,50]
sst_lonlim = [-180,180]
f_sst = '/gws/nopw/j04/ncas_climate_vol2/users/myoung02/datasets/SST/HadISST_sst.nc'
nc_fid = Dataset(f_sst, 'r')
sst_lat1 = np.array(nc_fid.variables['latitude'][:])  # extract/copy the data
sst_lon1 = np.array(nc_fid.variables['longitude'][:])
sst_time1 = np.array(nc_fid.variables['time'][:])
sst_time_units = str(np.array(nc_fid.variables['time'].units))
sst_dates = nc4.num2date(sst_time1,sst_time_units)
sst_time1 = np.nan*np.zeros((len(sst_dates),2))
for n in range(0,len(sst_dates)):
  sst_time1[n,:] = sst_dates[n].year,sst_dates[n].month
t_sst_id = np.where((sst_time1[:,0] >= years[0])&(sst_time1[:,0]<= years[-1]))[0]
sst_time = sst_time1[t_sst_id]
sst_lon_id = np.where((sst_lon1 >= sst_lonlim[0]) & (sst_lon1 <= sst_lonlim[1]))[0]
sst_lat_id = np.where((sst_lat1 >= sst_latlim[0]) & (sst_lat1 <= sst_latlim[1]))[0]
sst_lon = sst_lon1[sst_lon_id]
sst_lat = sst_lat1[sst_lat_id]
all_sst = np.array(nc_fid.variables['sst'][t_sst_id,:,:][:,sst_lat_id,:][:,:,sst_lon_id])
fill_sst = nc_fid.variables['sst']._FillValue  # shape is time, lat, lon as shown above
nc_fid.close()
all_sst[all_sst==fill_sst]=np.nan

all_sst = np.reshape(all_sst,(len(years),len(months),len(sst_lat),len(sst_lon)))
sst_time = np.reshape(sst_time[:,1],(len(years),len(months)))
sst_clim_mon = np.nanmean(all_sst,axis=0)
sst_anom = all_sst-np.repeat(sst_clim_mon[np.newaxis,:,:,:],len(years),axis=0)
  # sst_anom[m_id,:,:] = all_sst[m_id,:,:]-np.repeat(sst_clim_mon[m,np.newaxis,:,:],len(m_id),axis=0)

"""
SST REGIONS
For the Indian Ocean, we use the
standard definition of the Indian Ocean dipole (IOD;
averages of 10S-10N, 50-70E minus 10S-0, 90-110E; Saji et al. 1999), along with a central Indian Ocean index (CIndO; average of 25S-10N, 55-95E),
defined here because it correlates with the Sahel,
SEAfrica, and SWAfrica (see section 3a).
"""

n34_lonlim = [-170,-120]
n34_latlim = [-5,5]
iod1_lonlim = [50,70]
iod1_latlim = [-10,10]
iod2_lonlim = [90,110]
iod2_latlim = [-10,0]
io3_lonlim = [55,95]
io3_latlim = [-25,10]

siod1_lonlim = [55,65]
siod1_latlim = [-37,-27]
siod2_lonlim = [90,100]
siod2_latlim = [-28,-18]

siod1_lon_ind = np.where((sst_lon >= siod1_lonlim[0]) & (sst_lon <= siod1_lonlim[1]))
siod1_lat_ind = np.where((sst_lat >= siod1_latlim[0]) & (sst_lat <= siod1_latlim[1]))
sst_siod1 = sst_anom[:,:,siod1_lat_ind[0],:][:,:,:,siod1_lon_ind[0]]

siod2_lon_ind = np.where((sst_lon >= siod2_lonlim[0]) & (sst_lon <= siod2_lonlim[1]))
siod2_lat_ind = np.where((sst_lat >= siod2_latlim[0]) & (sst_lat <= siod2_latlim[1]))
sst_siod2 = sst_anom[:,:,siod2_lat_ind[0],:][:,:,:,siod2_lon_ind[0]]

iod1_lon_ind = np.where((sst_lon >= iod1_lonlim[0]) & (sst_lon <= iod1_lonlim[1]))
iod1_lat_ind = np.where((sst_lat >= iod1_latlim[0]) & (sst_lat <= iod1_latlim[1]))
sst_iod1 = sst_anom[:,:,iod1_lat_ind[0],:][:,:,:,iod1_lon_ind[0]]

iod2_lon_ind = np.where((sst_lon >= iod2_lonlim[0]) & (sst_lon <= iod2_lonlim[1]))
iod2_lat_ind = np.where((sst_lat >= iod2_latlim[0]) & (sst_lat <= iod2_latlim[1]))
sst_iod2 = sst_anom[:,:,iod2_lat_ind[0],:][:,:,:,iod2_lon_ind[0]]

n34_lon_ind = np.where((sst_lon >= n34_lonlim[0]) & (sst_lon <= n34_lonlim[1]))
n34_lat_ind = np.where((sst_lat >= n34_latlim[0]) & (sst_lat <= n34_latlim[1]))
sst_n34 = sst_anom[:,:,n34_lat_ind[0],:][:,:,:,n34_lon_ind[0]]

iod_index = np.nanmean(sst_iod1,axis=(2,3))-np.nanmean(sst_iod2,axis=(2,3))
siod_index = np.nanmean(sst_siod1,axis=(2,3))-np.nanmean(sst_siod2,axis=(2,3))
enso_index = np.nanmean(sst_n34,axis=(2,3))

'''
composite GPCC and TAMSAT and TAMSAT 3.1
on ENSO and IOD
'''
def running_mean(x, N):
  cumsum = np.cumsum(np.insert(x, 0, 0))
  return (cumsum[N:] - cumsum[:-N]) / N
# enso index is 3 month running mean?
# e_index_run = running_mean(enso_index.flatten(),3)
# e_index_run = np.reshape(e_index_run,(len(years),len(months)))
enso_index1= np.reshape(enso_index,len(years)*len(months))
sst_time2 = sst_time1[t_sst_id]
enso_threshold = 1 # k
el_events= np.where(enso_index1 > enso_threshold)[0]
la_events= np.where(enso_index1 < enso_threshold*-1)[0]
el_dates = sst_time2[el_events,:]
la_dates = sst_time2[la_events,:]

djf_id = np.sort(np.hstack((np.where((sst_time2[:,1] <3))[0],np.where((sst_time2[:,1]>10))[0])))
el_djf = np.intersect1d(djf_id,el_events)
la_djf = np.intersect1d(djf_id,la_events)

enso_gpcc = np.stack((np.nanmean(np.reshape(anom_gpcc,(len(years)*len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))[el_djf,:,:],axis=0),np.nanmean(np.reshape(anom_gpcc,(len(years)*len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))[la_djf,:,:],axis=0)))

enso_rfe = np.stack((np.nanmean(np.reshape(anom_rfe,(len(years)*len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))[el_djf,:,:],axis=0),np.nanmean(np.reshape(anom_rfe,(len(years)*len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))[la_djf,:,:],axis=0)))

enso_bias = np.stack((np.nanmean(np.reshape(all_bias,(len(years)*len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))[el_djf,:,:],axis=0),np.nanmean(np.reshape(all_bias,(len(years)*len(months),len(sub_gpcc_lat),len(sub_gpcc_lon)))[la_djf,:,:],axis=0)))

enso_all = np.stack((enso_gpcc,enso_rfe,enso_bias))

'''
Conditional bias on El Nino and La Nina events
'''
enso_label = ['El Nino','La Nina']
product_label = ['GPCC','TAMSATv3.1','Bias']
# Monthly mean bias
cmin = -50
cmax = 50
cspc = 10
clevs = np.arange(cmin,cmax+cspc,cspc)
norm = BoundaryNorm(boundaries=clevs, ncolors=256)
lw =1
cnt = 0
fname_plot = dir_out+'TAMSATv3.1_GPCC_DJF_ENSO_'+str(years[0])+'_'+str(years[-1])
fig = plt.figure(figsize=(7,5))
for e in [0,1]:
  for i in [0,1,2]:
    cnt = cnt + 1
    plt.subplot(2,3,cnt)
    mymap = Basemap(projection='cyl',resolution='l',\
          llcrnrlat=latlim[0],urcrnrlat=latlim[1],\
          llcrnrlon=lonlim[0],urcrnrlon=lonlim[1])
    mymap.drawcoastlines(linewidth=lw)
    x, y = mymap(*np.meshgrid(sub_gpcc_lon,sub_gpcc_lat))
    tmp_bias = []
    tmp_bias = np.copy(enso_all[i,e,:,:]).squeeze()
    tmp_bias[tmp_bias == 0] = np.nan
    p_anom = mymap.pcolormesh(x,y,enso_all[i,e,:,:],cmap='RdYlBu',vmin=cmin,vmax=cmax,norm=norm)
    if i == 0:
      plt.ylabel(enso_label[e],fontsize=12,fontweight='bold')
    if e == 0:
      plt.title(product_label[i])

fig.subplots_adjust(right=0.9)
# cbar_pos = [0.96, 0.5, 0.017, 0.3] #[left, bottom, width, height]
# cbar_ax = fig.add_axes(cbar_pos)
# cbar = fig.colorbar(p_anom,cax=cbar_ax,label='Anomaly (mm)',orientation='vertical',extend='both')
cbar_pos = [0.96, 0.1, 0.017, 0.35] #[left, bottom, width, height]
cbar_ax = fig.add_axes(cbar_pos)
cbar = fig.colorbar(p_bias,cax=cbar_ax,label='Anomaly/Bias (mm)',orientation='vertical',extend='both')
plt.tight_layout(pad=2.5,w_pad=0.02,h_pad=0.1)
plt.savefig(fname_plot+'.png',bbox_inches='tight',dpi=300)
plt.close()


'''
Bin era variables by TAMSATv3.1 bias
'''
bin_bias = np.arange(-500,500+100,100)
bias_era = np.nan*np.zeros((2,len(bin_bias),6))
bias_era_wind = np.nan*np.zeros((2,len(bin_bias),2,2))

flat_bias = all_bias.flatten()*rep_mask_gpcc.flatten()
flat_st = all_gpcc[1,:,:,:,:].flatten()
flat_era_w = anom_era_w.flatten()
flat_era_mf = anom_era_mf.flatten()
flat_era_u = np.reshape(anom_era_u,(2,len(flat_era_w)))
flat_era_v = np.reshape(anom_era_v,(2,len(flat_era_w)))
flat_era_rh = np.reshape(anom_era_rh,(2,len(flat_era_w)))
flat_era_vec = np.reshape(anom_era_vec,(2,len(flat_era_w)))
flat_era_shear = anom_era_shear.flatten()

for b in np.arange(0,len(bin_bias)):
  id_bias = []
  if b == 0:
    id_bias = np.where((flat_bias < bin_bias[b]))[0]
  elif (b>0) & (b<len(bin_bias)):
    id_bias = np.where((flat_bias >= bin_bias[b-1]) & (flat_bias < bin_bias[b]))[0]
  elif b == len(bin_bias):
    id_bias = np.where((flat_bias >= bin_bias[b]))[0]
  print len(id_bias)
  id_st = []
  id_st = np.where(flat_st > 0)[0]
  id_val = []
  id_val = np.intersect1d(id_bias,id_st)

  bias_era[:,b,0] = np.nanmean(flat_era_w[id_val]),np.nanstd(flat_era_w[id_val])/(np.sqrt(len(id_val)))
  bias_era[:,b,1] = np.nanmean(flat_era_shear[id_val]),np.nanstd(flat_era_shear[id_val])/(np.sqrt(len(id_val)))
  bias_era[:,b,2] = np.nanmean(flat_era_rh[0,id_val]),np.nanstd(flat_era_rh[0,id_val])/(np.sqrt(len(id_val)))
  bias_era[:,b,3] = np.nanmean(flat_era_rh[1,id_val]),np.nanstd(flat_era_rh[1,id_val])/(np.sqrt(len(id_val)))
  bias_era[:,b,4] = np.nanmean(flat_era_mf[id_val]),np.nanstd(flat_era_mf[id_val])/(np.sqrt(len(id_val)))
  bias_era[:,b,5] = len(id_val)

  bias_era_wind[:,b,0,0] = np.nanmean(flat_era_u[0,id_val]),np.nanstd(flat_era_u[0,id_val])/(np.sqrt(len(id_val)))
  bias_era_wind[:,b,0,1] = np.nanmean(flat_era_u[1,id_val]),np.nanstd(flat_era_u[1,id_val])/(np.sqrt(len(id_val)))
  bias_era_wind[:,b,1,0] = np.nanmean(flat_era_v[0,id_val]),np.nanstd(flat_era_v[0,id_val])/(np.sqrt(len(id_val)))
  bias_era_wind[:,b,1,1] = np.nanmean(flat_era_v[1,id_val]),np.nanstd(flat_era_v[1,id_val])/(np.sqrt(len(id_val)))

# plot bias vs diagnostic
era_short_name= ['w_500','shear','rh_250','rh_850','vimd']
era_metric_ls = ['500 hPa $\omega$ (Pa s$^{-1}$)','250 hPa - 850 hPa shear (ms$^{-1}$)','250 hPa R.H. (%)','850 hPa R.H. (%)','Vert. Int. Moisture Div. (kg m$^{-2}$)']
for p in np.arange(0,5):
  fname_plot = dir_out+'ERA5_TAMSATv3.1_GPCC_monthly_bias_vs_'+era_short_name[p]+'_'+str(years[0])+'_'+str(years[-1])
  plt.figure(figsize=(3.5,3))
  plt.errorbar(bin_bias,bias_era[0,:,p],yerr=bias_era[1,:,p]*0.5,color='black',linewidth =2,marker='o', markersize=8)
  plt.axvline(x=0,linewidth=1, color='grey')
  plt.axhline(y=0,linewidth=1, color='grey')
  plt.ylabel('Anomalous \n'+era_metric_ls[p])
  plt.xlabel('Bias (mm)')
  plt.tight_layout()
  plt.savefig(fname_plot+'.png',bbox_inches='tight',dpi=300)
  plt.close()

era_p_ls = ['250 hPa','850 hPa']
era_wind_ls =['U-wind (m s$^{-1}$)','V-wind (m s$^{-1}$)']
for p in [0,1]:
  for l in [0,1]:
    plt.figure(figsize=(3,3))
    plt.errorbar(bin_bias,bias_era_wind[0,:,p,l],yerr=bias_era_wind[1,:,p,l]*0.5,color='black',linewidth =2,marker='o', markersize=8)
    plt.axvline(x=0,linewidth=1, color='grey')
    plt.axhline(y=0,linewidth=1, color='grey')
    plt.ylabel('Anomalous \n'+era_p_ls[l]+' '+era_wind_ls[p])
    plt.xlabel('Bias (mm)')
plt.show()


# do monthly bias vs diganostic
bias_era_month = np.nan*np.zeros((12,2,len(bin_bias),6))

for m in np.arange(0,12):
  flat_bias = all_bias[:,m,:,:].flatten()*rep_mask_gpcc[:,m,:,:].flatten()
  flat_st = all_gpcc[1,:,m,:,:].flatten()
  flat_era_w = anom_era_w[:,m,:,:].flatten()
  flat_era_mf = anom_era_mf[:,m,:,:].flatten()
  flat_era_u = np.reshape(anom_era_u[:,:,m,:,:],(2,len(flat_era_w)))
  flat_era_v = np.reshape(anom_era_v[:,:,m,:,:],(2,len(flat_era_w)))
  flat_era_rh = np.reshape(anom_era_rh[:,:,m,:,:],(2,len(flat_era_w)))
  flat_era_vec = np.reshape(anom_era_vec[:,:,m,:,:],(2,len(flat_era_w)))
  flat_era_shear = anom_era_shear[:,m,:,:].flatten()

  for b in np.arange(0,len(bin_bias)):
    id_bias = []
    if b == 0:
      id_bias = np.where((flat_bias < bin_bias[b]))[0]
    elif (b>0) & (b<len(bin_bias)):
      id_bias = np.where((flat_bias >= bin_bias[b-1]) & (flat_bias < bin_bias[b]))[0]
    elif b == len(bin_bias):
      id_bias = np.where((flat_bias >= bin_bias[b]))[0]
    print len(id_bias)
    id_st = []
    id_st = np.where(flat_st > 0)[0]
    id_val = []
    id_val = np.intersect1d(id_bias,id_st)

    bias_era_month[m,:,b,0] = np.nanmean(flat_era_w[id_val]),np.nanstd(flat_era_w[id_val])/(np.sqrt(len(id_val)))
    bias_era_month[m,:,b,1] = np.nanmean(flat_era_shear[id_val]),np.nanstd(flat_era_shear[id_val])/(np.sqrt(len(id_val)))
    bias_era_month[m,:,b,2] = np.nanmean(flat_era_rh[0,id_val]),np.nanstd(flat_era_rh[0,id_val])/(np.sqrt(len(id_val)))
    bias_era_month[m,:,b,3] = np.nanmean(flat_era_rh[1,id_val]),np.nanstd(flat_era_rh[1,id_val])/(np.sqrt(len(id_val)))
    bias_era_month[m,:,b,4] = np.nanmean(flat_era_mf[id_val]),np.nanstd(flat_era_mf[id_val])/(np.sqrt(len(id_val)))
    bias_era_month[m,:,b,5] = len(id_val)

# plot monthly bias vs diagnostic
col_ls = cm.rainbow(np.linspace(0,1,12))

era_short_name= ['w_500','shear','rh_250','rh_850','vimd']
era_metric_ls = ['500 hPa $\omega$ (Pa s$^{-1}$)','250 hPa - 850 hPa shear (ms$^{-1}$)','250 hPa R.H. (%)','850 hPa R.H. (%)','Vert. Int. Moisture Div. (kg m$^{-2}$)']
for p in np.arange(0,5):
  fname_plot = dir_out+'ERA5_TAMSATv3.1_GPCC_monthly_month_bias_vs_'+era_short_name[p]+'_'+str(years[0])+'_'+str(years[-1])
  plt.figure(figsize=(4.5,3))
  for m in np.arange(0,12):
    plt.errorbar(bin_bias,bias_era_month[m,0,:,p],yerr=bias_era_month[m,1,:,p]*0.5,color=col_ls[m,:],linewidth =2,marker='o', markersize=8,alpha=0.6,label=str(m+1))
  plt.axvline(x=0,linewidth=1, color='grey')
  plt.axhline(y=0,linewidth=1, color='grey')
  plt.ylabel('Anomalous \n'+era_metric_ls[p])
  plt.xlabel('Bias (mm)')
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 10})
  plt.tight_layout()
  # plt.show()
  plt.savefig(fname_plot+'.png',bbox_inches='tight',dpi=300)
  plt.close()


'''
Plot large scale regime associated with regional biases
(Haven't coded yet, but idea is to select a region and
compute timeseries of the mean bias over that region
and create spatial composites of ERA5 fields for months when
the mean bias over the specific region exceeds a threshold

e.g. Composite ERA5 fields during months when TAMSATv3.1 bias over
East Africa is < - 5 mm/d

Commented below are some example regions which could be used
'''

# # monthly time series of bias over africa regions
# region_ls = ['Central Africa','East Africa','West Africa','Southern Africa']
# for region in [0,1,2,3]:
#   if region == 0:
#     region_name='Central Africa'
#     reg_stamp = 'CAfrica'
#     lonlim_region = [10,30]
#     latlim_region = [-10,5]
#     lonlim_region1= [0,40]
#     latlim_region1 = [-15,10]
#     r_bias =[1.5,-0.5]
#     size2 = [10,8]
#   elif region == 1:
#     region_name='East Africa'
#     reg_stamp = 'EAfrica'
#     lonlim_region = [35,45]
#     latlim_region = [-10,10]
#     lonlim_region1= [30,50]
#     latlim_region1 = [-15,15]
#     r_bias =[0.5,-0.5]
#     size2 = [10,12]
#   elif region == 2:
#     region_name='West Africa'
#     reg_stamp = 'WAfrica'
#     lonlim_region = [-17,10]
#     latlim_region = [4,10]
#     lonlim_region1= [-20,20]
#     latlim_region1 = [0,20]
#     r_bias =[1,-0.5]
#     size2 = [10,8]
#   elif region == 3:
#     region_name='Southern Africa'
#     reg_stamp = 'SAfrica'
#     lonlim_region = [15,35]
#     latlim_region = [-35,-20]
#     lonlim_region1= [5,52]
#     latlim_region1 = [-35,-10]
#     r_bias =[0.5,-0.5]
#     size2 = [10,7]
