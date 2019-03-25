#!/bin/bash
for i in {1..10}; do nohup ./faraday.py --geometry box --xmesh 20 --ymesh 5 --zmesh 10 --width 1 --length 4 --height 2 --iamp 5e-3 --bmesh 0 --nonlinear 1 --time 200 --filebase data/box3saturate$i --contact slip  --frequency $((20+i)) --acceleration 0.7 &> outs/box3saturate${i}.out & done
