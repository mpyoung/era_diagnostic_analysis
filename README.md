# era_diagnostic_analysis

M. Young April 2020

## Main analysis script:
 - analyse_bias_era5.py

## Other scripts:
 - date_str.py : some functions that sort out date strings
 - grab_gpcc.py : functions for reading in GPCC monitoring product



## Data (ERA5) Download scripts
Python scripts that download ERA5 hourly/monthly fields from copernicus using the cds api
 - get_era5_hourly.py
 - get_era5_monthly.py

Shell scripts that submit the above python download scripts to run on jasmin/lotus
 - submit_bjob_dl_era5_hrly.sh
 - submit_bjob_dl_era5_monthly.sh

## Data re-grid scripts
Python scripts to re-grid ERA5 and TAMSATv3.1 data to GPCC grid
 - regrid_era5_hourly.py
 - regrid_era5_monthly.py
 - regrid_tamsat31_monthly.py

Shell scripts that submit the above python re-grid scripts to run on jasmin/lotus
 - submit_bjob_regrid_era5_hrly.sh
 - submit_bjob_regrid_era5_monthly.sh
 - submit_bjob_regrid_tamsat31_monthly.sh
