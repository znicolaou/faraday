# Files in the faraday repository
The file faraday.py contains python code to integrate the equations of motion for inviscid Faraday waves in inhomogeneous domains with the finite element method. The files plotrectangle.nb, plotcylinder.nb, and plotbox.nb are Mathematica notebooks to plot results from simulations.

# System requirements
The python code has been run with anaconda 2.7, which can be downloaded here: https://www.anaconda.com/distribution/.  The script requires packes numpy, scipy, fenics-2018.1.0, and mshr-2018.1.0, which can be installed after installing anaconda with the shell command `conda install -c conda-forge fenics=2018.1.0 mshr=2018.1.0 numpy scipy`

# Usage
Running the script `./faraday.py -h` will produce the following usage message:

```
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

Moving mesh simulation for inviscid Faraday waves with inhomogeneous  
substrate.

optional arguments:
  -h, --help            show this help message and exit
  --frequency FREQ      Driving frequency in Hertz
  --gravity G           Gravitational acceleration in cm/s^2
  --acceleration ACCELERATION
                        Driving acceleration in terms of gravitational
                        acceleration
  --width WIDTH         Width in cm
  --length LENGTH       Length in cm
  --height HEIGHT       Height in cm
  --radius RADIUS       Radius in cm
  --tension SIGMA       Surface tension in dyne/cm^2
  --density RHO         Fluid density in g/cm^3
  --time SIMTIME        Simulation time in driving cycles
  --steps STEPS         Output steps per cycle
  --output {0,1}        Flag to output full data or abbreviated data
  --iseed ISEED         Seed for random initial conditions
  --iamp IAMP           Amplitude for modes in random initial conditions
  --imodes IMODES       Number of modes to include in random initial
                        conditions
  --sseed SSEED         Seed for random substrate shape
  --samp SAMP           Amplitude for modes in random substrate shape
  --smodes SMODES       Number of modes to include in random substrate shape
  --rtol RTOL           Integration relative tolerance
  --atol ATOL           Integration absolute tolerance
  --damp1 DAMP1         Constant damping coefficient
  --damp3 DAMP2         Curvature damping coefficient
  --xmesh XMESH         Lateral mesh refinement
  --ymesh YMESH         Lateral mesh refinement
  --zmesh ZMESH         Vertical mesh refinement
  --threshold THRS      Threshold change in log norm magnitude to stop
                        integration
  --filebase OUTPUT     Base string for file output
  --refinement REFINEMENT
                        Number of refinements for top
  --bmesh {0,1}         Flag to move boundary mesh in time stepping. This is
                        faster and tentatively more accurate than the
                        alternative mesh movement, but suffers numerical
                        instabilities for large deviations.
  --nonlinear {0,1}     Flag to include nonlinear terms
  --geometry {rectangle,cylinder,box}
                        Mesh geometry. Options are rectangle, cylinder, and
                        box.
  --contact {stick,slip,periodic}
                        Contact line boundary conditions. Options are stick,
                        slip, and periodic. periodic is not available for
                        cylinder geometry.
```

