#!/bin/bash
export err_dir=/home/users/myoung02/code/python/era_diagnostic_analysis/bsub_output/

for y in $(seq 1983 2019); do
  echo "
  #!/bin/bash
  #BSUB -o ${err_dir}%J.o
  #BSUB -e ${err_dir}%J.e
  #BSUB -q short-serial
  #BSUB -J matt_py_job
  #BSUB -n 1
  #BSUB -W 1:00
  #BSUB -M 40000
  python regrid_tamsat31_monthly.py $y 2>> ${err_dir}err_t3${y}_${i}.log >> ${err_dir}out_t3${y}_${i}.log
  " >> job_rg_t3${y}_${i}.sh
  bsub < job_rg_t3${y}_${i}.sh
  rm job_rg_t3${y}_${i}.sh
done
