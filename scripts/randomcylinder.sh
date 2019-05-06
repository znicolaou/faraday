#!/bin/bash
#MSUB -A p30575
#MSUB -q normal
#MSUB -l procs=25
#MSUB -j oe
#MSUB -V
#MSUB -o "$HOME/faraday/outs/rc1_$MOAB_JOBARRAYINDEX.out"
cd $HOME/faraday/
source activate fenics_env

ZGN_filebase=$HOME/faraday/rc1

ZGN_freq0=8.0
ZGN_freq1=21.0
ZGN_N=50
ZGN_xmesh=15
ZGN_ymesh=5
ZGN_zmesh=10
ZGN_samp=1.75
ZGN_smodes=8
ZGN_iamp=1e-4
ZGN_contact="stick"
ZGN_geometry="cylinder"
ZGN_threshold=2
ZGN_height=2
ZGN_length=4
ZGN_width=4
ZGN_height=2
ZGN_radius=2
ZGN_time=50
ZGN_A=0.6
ZGN_temp2=$MOAB_JOBARRAYINDEX
if [ -d $ZGN_filebase/${ZGN_temp2} ]; then rm -r $ZGN_filebase/${ZGN_temp2}; fi
mkdir -p $ZGN_filebase/${ZGN_temp2}
if [ $ZGN_temp2 -eq 1 ]; then ZGN_samp=0; fi
for ZGN_temp1 in `seq 1 $ZGN_N`; do

js=`jobs | wc -l`
while [ $js -ge 20 ]; do
sleep 1
js=`jobs | wc -l`
done

ZGN_freq=`bc -l <<<"${ZGN_freq0}+(${ZGN_freq1}-${ZGN_freq0})*${ZGN_temp1}*1.0/${ZGN_N}"`
ZGN_sseed=$ZGN_temp2

echo "/home/zgn667/anaconda3/envs/fenics_env/bin/python faraday.py --frequency $ZGN_freq --acceleration $ZGN_A --filebase ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1} --output 0 --iamp $ZGN_iamp --xmesh $ZGN_xmesh --ymesh $ZGN_ymesh --zmesh $ZGN_zmesh --samp $ZGN_samp --threshold $ZGN_threshold --contact $ZGN_contact --height $ZGN_height --width $ZGN_width --length $ZGN_length --radius $ZGN_radius --geometry $ZGN_geometry --time $ZGN_time --sseed $ZGN_sseed --smodes $ZGN_smodes &"
/home/zgn667/anaconda3/envs/fenics_env/bin/python faraday.py --frequency $ZGN_freq --acceleration $ZGN_A --filebase ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1} --output 0 --iamp $ZGN_iamp --xmesh $ZGN_xmesh --ymesh $ZGN_ymesh --zmesh $ZGN_zmesh --samp $ZGN_samp --threshold $ZGN_threshold --contact $ZGN_contact --height $ZGN_height --width $ZGN_width --length $ZGN_length --radius $ZGN_radius --geometry $ZGN_geometry --time $ZGN_time --sseed $ZGN_sseed --smodes $ZGN_smodes &
done

wait

for ZGN_temp1 in `seq 1 $ZGN_N`; do echo $ZGN_temp1; cat ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1}.txt >> ${ZGN_filebase}/params${ZGN_temp2}.txt; rm ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1}.txt; done
rmdir ${ZGN_filebase}/${ZGN_temp2}
source deactivate