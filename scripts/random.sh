#!/bin/bash
#SBATCH -A p30471
#SBATCH -n 50
#SBATCH -p normal
#SBATCH -t 48:00:00
#SBATCH --mem=7500
#SBATCH --array=0-500
#SBATCH --output=outs/random_%a.out
cd $HOME/faraday/
source activate fenics_env

ZGN_filebase=$HOME/faraday/random

ZGN_freq0=2.0
ZGN_freq1=30.0
ZGN_N=100
ZGN_xmesh=50
ZGN_ymesh=5
ZGN_zmesh=5
ZGN_samp=0.4
ZGN_smodes=12
ZGN_iamp=1e-4
ZGN_contact="stick"
ZGN_geometry="rectangle"
ZGN_threshold=2
ZGN_height=0.5
ZGN_length=3.5
ZGN_time=50
ZGN_A=0.8
ZGN_damp1=2.0
ZGN_damp2=0.1
ZGN_temp2=$SLURM_ARRAY_TASK_ID
if [ -d $ZGN_filebase/${ZGN_temp2} ]; then rm -r $ZGN_filebase/${ZGN_temp2}; fi
mkdir -p $ZGN_filebase/${ZGN_temp2}
if [ $ZGN_temp2 -eq 1 ]; then ZGN_samp=0; fi
for ZGN_temp1 in `seq 1 $ZGN_N`; do

js=`jobs | wc -l`
while [ $js -ge 50 ]; do
sleep 1
js=`jobs | wc -l`
done

ZGN_freq=`bc -l <<<"${ZGN_freq0}+(${ZGN_freq1}-${ZGN_freq0})*${ZGN_temp1}*1.0/${ZGN_N}"`
ZGN_sseed=$ZGN_temp2

echo "python faraday.py --frequency $ZGN_freq --acceleration $ZGN_A --filebase ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1} --output 0 --iamp $ZGN_iamp --xmesh $ZGN_xmesh --zmesh $ZGN_zmesh --threshold $ZGN_threshold --contact $ZGN_contact --height $ZGN_height --length $ZGN_length --geometry $ZGN_geometry --time $ZGN_time --samp $ZGN_samp --sseed $ZGN_sseed --smodes $ZGN_smodes --damp1 $ZGN_damp1 --damp2 $ZGN_damp2 &"
python faraday.py --frequency $ZGN_freq --acceleration $ZGN_A --filebase ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1} --output 0 --iamp $ZGN_iamp --xmesh $ZGN_xmesh --zmesh $ZGN_zmesh --threshold $ZGN_threshold --contact $ZGN_contact --height $ZGN_height --length $ZGN_length --geometry $ZGN_geometry --time $ZGN_time --samp $ZGN_samp --sseed $ZGN_sseed --smodes $ZGN_smodes --damp1 $ZGN_damp1 --damp2 $ZGN_damp2 &

done

wait

for ZGN_temp1 in `seq 1 $ZGN_N`; do echo $ZGN_temp1; cat ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1}.txt >> ${ZGN_filebase}/params${ZGN_temp2}.txt; rm ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1}.txt; done
rmdir ${ZGN_filebase}/${ZGN_temp2}
source deactivate
