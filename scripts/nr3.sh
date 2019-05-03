#!/bin/bash
#MSUB -A p30575
#MSUB -q short
#MSUB -l procs=15
#MSUB -t [1-100]
#MSUB -j oe
#MSUB -V
#MSUB -o "$HOME/faraday/outs/nr3_$MOAB_JOBARRAYINDEX.out"
cd $HOME/faraday/
source activate fenics_env

ZGN_filebase=$HOME/faraday/nr3

ZGN_freq0=2
ZGN_freq1=30
#ZGN_freq=18
ZGN_samp0=0.0
ZGN_samp1=1.75
ZGN_samp=1.75
ZGN_A0=0.2
ZGN_A1=2.0
ZGN_A=0.8
ZGN_N1=90
ZGN_N2=$MOAB_JOBARRAYRANGE
ZGN_xmesh=50
ZGN_ymesh=5
ZGN_zmesh=10
ZGN_iamp=1e-4
ZGN_contact="stick"
ZGN_geometry="rectangle"
ZGN_threshold=2
ZGN_height=2
ZGN_length=4
ZGN_width=4
ZGN_time=100
ZGN_sseed=47
ZGN_smodes=8
ZGN_temp2=$MOAB_JOBARRAYINDEX
if [ -d $ZGN_filebase/${ZGN_temp2} ]; then rm -r $ZGN_filebase/${ZGN_temp2}; fi
mkdir -p $ZGN_filebase/${ZGN_temp2}

for ZGN_temp1 in `seq 1 $ZGN_N1`; do

cp $ZGN_filebase/substrate.dat $ZGN_filebase/${ZGN_temp2}/params_${ZGN_temp1}substrate.dat

js=`jobs | wc -l`
while [ $js -ge $PBS_NP ]; do
sleep 1
js=`jobs | wc -l`
done

ZGN_A=`bc -l <<<"${ZGN_A0}+(${ZGN_A1}-${ZGN_A0})*$((ZGN_temp1-1))*1.0/$((ZGN_N1-1))"`
#ZGN_samp=`bc -l <<<"${ZGN_samp0}+(${ZGN_samp1}-${ZGN_samp0})*$((ZGN_temp1-1))*1.0/$((ZGN_N1-1))"`
ZGN_freq=`bc -l <<<"${ZGN_freq0}+(${ZGN_freq1}-${ZGN_freq0})*$((ZGN_temp2-1))*1.0/$((ZGN_N2-1))"`

/home/zgn667/anaconda3/envs/fenics_env/bin/python faraday.py --frequency $ZGN_freq --acceleration $ZGN_A --filebase ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1} --output 0 --iamp $ZGN_iamp --xmesh $ZGN_xmesh --ymesh $ZGN_ymesh --zmesh $ZGN_zmesh --samp $ZGN_samp --threshold $ZGN_threshold --contact $ZGN_contact --height $ZGN_height --width $ZGN_width --length $ZGN_length --geometry $ZGN_geometry --time $ZGN_time --sseed $ZGN_sseed --smodes $ZGN_smodes &
done

wait

for ZGN_temp1 in `seq 1 $ZGN_N1`; do 
echo $ZGN_temp1; cat ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1}.txt >> ${ZGN_filebase}/params${ZGN_temp2}.txt 
rm ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1}.txt
rm ${ZGN_filebase}/${ZGN_temp2}/params_${ZGN_temp1}substrate.dat
done
rmdir ${ZGN_filebase}/${ZGN_temp2}
source deactivate
