#!/bin/bash
for i in {1..5}; do nohup ./faraday.py --geometry box --xmesh 8 --ymesh 3 --zmesh 5 --refinement 1 --width 1 --length 4 --height 2 --iamp 5e-3 --bmesh 0 --nonlinear 1 --time 200 --filebase data/boxsaturate$i --contact periodic  --frequency $((21+i)) --acceleration 1.6 &> outs/boxsaturate${i}.out & done
