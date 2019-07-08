#!/bin/bash
#SBATCH -A p30471
#SBATCH -N 2
#SBATCH -n 56
#SBATCH -p short
#SBATCH -t 04:00:00
#SBATCH --mem=5000
#SBATCH --array=1-56
#SBATCH --output=outs/random2_%a.out
cd ~/faraday/
source activate fenics_env

ZGN_filebase=~/faraday/random2

ZGN_freq0=8
ZGN_freq1=35
#ZGN_freq=18
ZGN_samp0=0.0
ZGN_samp1=0.4
ZGN_samp=0.0
ZGN_A0=0.0
ZGN_A1=0.5
# ZGN_A=0.8
ZGN_tension=72
ZGN_N1=56
ZGN_N2=56
ZGN_xmesh=50
ZGN_zmesh=10
ZGN_iamp=1e-4
ZGN_contact="stick"
ZGN_geometry="rectangle"
ZGN_threshold=2
ZGN_height=0.5
ZGN_length=3.5
ZGN_time=100
ZGN_temp2=$SLURM_ARRAY_TASK_ID
ZGN_samp=0.4
ZGN_sseed=191
ZGN_smodes=12
ZGN_damp1=2.0
ZGN_damp2=0.1
if [ -d $ZGN_filebase/${ZGN_temp2} ]; then rm -r $ZGN_filebase/${ZGN_temp2}; fi
mkdir -p $ZGN_filebase/${ZGN_temp2}

for ZGN_temp1 in `seq 1 $ZGN_N1`; do

js=`jobs | wc -l`
while [ $js -ge $SLURM_NTASKS ]; do
sleep 1
js=`jobs | wc -l`
done

ZGN_Ad=`bc -l <<<"${ZGN_A0}+(${ZGN_A1}-${ZGN_A0})*$((ZGN_temp1-1))*1.0/$((ZGN_N1-1))"`
#ZGN_samp=`bc -l <<<"${ZGN_samp0}+(${ZGN_samp1}-${ZGN_samp0})*$((ZGN_temp1-1))*1.0/$((ZGN_N1-1))"`
ZGN_freq=`bc -l <<<"${ZGN_freq0}+(${ZGN_freq1}-${ZGN_freq0})*$((ZGN_temp2-1))*1.0/$((ZGN_N2-1))"`
ZGN_A=`bc -l <<<"${ZGN_Ad}/980*(2*3.14159*${ZGN_freq})^2"`

echo "python faraday.py --frequency $ZGN_freq --acceleration $ZGN_A --filebase ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1} --output 0 --iamp $ZGN_iamp --xmesh $ZGN_xmesh --zmesh $ZGN_zmesh --threshold $ZGN_threshold --contact $ZGN_contact --height $ZGN_height --length $ZGN_length --geometry $ZGN_geometry --tension $ZGN_tension --time $ZGN_time --samp $ZGN_samp --sseed $ZGN_sseed --smodes $ZGN_smodes &"

python faraday.py --frequency $ZGN_freq --acceleration $ZGN_A --filebase ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1} --output 0 --iamp $ZGN_iamp --xmesh $ZGN_xmesh --zmesh $ZGN_zmesh --threshold $ZGN_threshold --contact $ZGN_contact --height $ZGN_height --length $ZGN_length --geometry $ZGN_geometry --tension $ZGN_tension --time $ZGN_time --samp $ZGN_samp --sseed $ZGN_sseed --smodes $ZGN_smodes --damp1 $ZGN_damp1 --damp2 $ZGN_damp2 &
done

wait

for ZGN_temp1 in `seq 1 $ZGN_N1`; do
echo $ZGN_temp1; cat ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1}.txt >> ${ZGN_filebase}/params${ZGN_temp2}.txt
rm ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1}.txt
done
rmdir ${ZGN_filebase}/${ZGN_temp2}
source deactivate
