#!/bin/bash
for i in {1..5}; do nohup ./faraday.py --geometry box --xmesh 20 --ymesh 5 --zmesh 10 --width 1 --length 4 --height 2 --iamp 5e-3 --bmesh 0 --nonlinear 1 --time 200 --filebase data/box2saturate$i --contact stick  --frequency $((20+i)) --acceleration 1.6 &> outs/box2saturate${i}.out & done
