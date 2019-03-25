#!/bin/bash
for i in {1..10}; do nohup ./faraday.py --geometry cylinder --xmesh 15 --zmesh 15 --refinement 0 --radius 2 --height 2 --iamp 5e-3 --bmesh 0 --nonlinear 1 --time 200 --filebase data/cylinder2saturate${i} --contact slip  --frequency $((20+i)) --acceleration 0.7 &> outs/cylinder2saturate${i}.out & done
