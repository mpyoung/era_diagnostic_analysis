#!/bin/bash
export err_dir=/home/users/myoung02/code/python/era_diagnostic_analysis/bsub_output/
export var_name='vertically_integrated_moisture_divergence'

for m in $(seq 1 1);do
  for y in $(seq 1986 1986); do
    echo "
    #!/bin/bash
    #BSUB -o ${err_dir}%J.o
    #BSUB -e ${err_dir}%J.e
    #BSUB -q short-serial
    #BSUB -J matt_py_job
    #BSUB -n 1
    #BSUB -W 1:00
    #BSUB -M 40000
    python get_era5_hourly.py $y $m ${var_name[$i]} 2>> ${err_dir}err_e5_hr_${y}_${i}.log >> ${err_dir}out_e5_hr_${y}_${i}.log
    " >> job_dl_e5_hr_${y}_${i}.sh
    bsub < job_dl_e5_hr_${y}_${i}.sh
    rm job_dl_e5_hr_${y}_${i}.sh
  done
done
