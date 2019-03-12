#!/bin/bash
for i in {1..5}; do nohup ./faraday.py --geometry cylinder --xmesh 15 --zmesh 15 --refinement 0 --radius 2 --height 2 --iamp 5e-3 --bmesh 0 --nonlinear 1 --time 200 --filebase data/cylindersaturate${i} --contact stick  --frequency $((20+i)) --acceleration 1.0 &> outs/cylindersaturate${i}.out & done
