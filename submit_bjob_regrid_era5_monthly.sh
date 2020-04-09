#!/bin/bash
export err_dir=/home/users/myoung02/code/python/era_diagnostic_analysis/bsub_output/
export var_name=('relative_humidity' 'relative_humidity' 'u_component_of_wind' 'u_component_of_wind' 'v_component_of_wind' 'v_component_of_wind' 'vertical_velocity')
export p_lev=('250' '850' '250' '850' '250' '850' '500')

for i in $(seq 0 6); do
  for y in $(seq 2019 2019); do
    echo "
    #!/bin/bash
    #BSUB -o ${err_dir}%J.o
    #BSUB -e ${err_dir}%J.e
    #BSUB -q short-serial
    #BSUB -J matt_py_job
    #BSUB -n 1
    #BSUB -W 1:00
    #BSUB -M 40000
    python regrid_era5_monthly.py $y ${var_name[$i]} ${p_lev[$i]} 2>> ${err_dir}err_e5${y}_${i}.log >> ${err_dir}out_e5${y}_${i}.log
    " >> job_dl_e5${y}_${i}.sh
    bsub < job_dl_e5${y}_${i}.sh
    rm job_dl_e5${y}_${i}.sh
  done
done
