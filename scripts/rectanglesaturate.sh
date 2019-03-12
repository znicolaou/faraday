#!/bin/bash
for i in {1..5}; do nohup ./faraday.py --geometry rectangle --xmesh 25 --zmesh 10 --refinement 1 --width 4 --height 2 --iamp 5e-3 --bmesh 0 --nonlinear 1 --time 200 --filebase data/rectanglesaturate$i --contact periodic  --frequency $((20+i)) --acceleration 0.6 &> outs/rectanglesaturate${i}.out & done
