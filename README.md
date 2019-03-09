# Files in the faraday repository
The file faraday.py contains python code to integrate the equations of motion for inviscid Faraday waves in inhomogeneous domains with the finite element method. The files plotrectangle.nb, plotcylinder.nb, and plotbox.nb are Mathematica notebooks to plot results from simulations.

# System requirements
The python code has been run with anaconda 2.7, which can be downloaded here: https://www.anaconda.com/distribution/.  The script requires packes numpy, scipy, fenics-2018.1.0, and mshr-2018.1.0, which can be installed after installing anaconda with the shell command `conda install -c conda-forge fenics=2018.1.0 mshr=2018.1.0 numpy scipy`

# Usage
Running the script `./faraday.py` will produce the following usage message:

usage: faraday.py [-h] [--frequency FREQ] [--gravity G]  
                  [--acceleration ACCELERATION] [--width WIDTH]  
                  [--length LENGTH] [--height HEIGHT] [--radius RADIUS]  
                  [--tension SIGMA] [--density RHO] [--time SIMTIME]  
                  [--steps STEPS] [--output {0,1}] [--iseed ISEED]  
                  [--iamp IAMP] [--imodes IMODES] [--sseed SSEED]  
                  [--samp SAMP] [--smodes SMODES] [--rtol RTOL] [--atol ATOL]  
                  [--damp1 DAMP1] [--damp3 DAMP2] [--xmesh XMESH]  
                  [--ymesh YMESH] [--zmesh ZMESH] [--threshold THRS]  
                  --filebase OUTPUT [--refinement REFINEMENT] [--bmesh {0,1}]  
                  [--nonlinear {0,1}] [--geometry {rectangle,cylinder,box}]  
                  [--contact {stick,slip,periodic}]  
faraday.py: error: the following arguments are required: --filebase  
