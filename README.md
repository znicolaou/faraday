# Files in the faraday repository
The file faraday.py is a python script to integrate the equations of motion for inviscid Faraday waves in inhomogeneous domains with the finite element method. The files plotrectangle.nb, plotcylinder.nb, and plotbox.nb are Mathematica notebooks to plot results from simulations. The file accel.py is a python script for controlling an Arduino to collect data from an accelerometer in Faraday wave experiments.  The arduino-cli files are command line interfaces for controlling the Arduino.

# System requirements
The python code has been run with Anaconda 2.7, which can be downloaded here: https://www.anaconda.com/distribution/.  The finite element method uses the [FEniCS Project](https://fenicsproject.org/) implementation, while the time integration uses [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html) implementation of the Real-valued Variable-coefficient Ordinary Differential Equation solver. The scripts require packages numpy, scipy, matplotlib, pyserial, fenics-2018.1.0, and mshr-2018.1.0, Create a new anaconda environment and install from the default channels with `conda create -n fenics_env numpy scipy matplotlib pyserial`.   Activate the environment with `conda activate faraday_env`, then install fenics and mshr from the conda-forge channel with `conda install -c conda-forge fenics=2018.1.0 mshr=2018.1.0`.  NB the matplotlib package from the conda-forge channel has not worked well on mac computers.

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
                  [--contact {stick,slip,periodic}] [--nthreads NTHREADS]

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
  --nthreads NTHREADS   Number of threads to allow parallel computations to
                        run over.
```

Running the script `./accel.py -h` will produce the following usage message:  

```
usage: accel.py [-h] --filebase FILEBASE [--directory DATA] [--Nt NT]  
                [--delay DELAY] [--xy {0,1}] [--run {0,1}] [--count COUNT]  

Upload an Arduino sketch and read output from the accelerometer.  

optional arguments:  
  -h, --help           show this help message and exit  
  --filebase FILEBASE  Base string for file output.  
  --directory DATA     Directory to save files. Default "data".  
  --Nt NT              Number of buffer ints. Default 150.  
  --delay DELAY        Delay between samples. Default 2.0.  
  --xy {0,1}           Flag for x and y output. Default 1.  
  --run {0,1}          Flag for running arduino and reading output; if 0, data  
                       is read from previous runs if files exist. Default 1.  
  --count COUNT        Initial count. Default 0.  
  ```

# Input file formats
For faraday.py script, if the file filebasesubstrate.dat exists, the script reads a line containing an integer Nmodes giving the number of modes, followed by lines which each contain Nmodes numbers specifying the sine and cosine mode amplitudes for the substrate shape. For rectangle geometries, there is a single sine and a single cosine line.  For the box geometries, there are Nmodes lines for sine-sine modes, followed by Nmodes lines for sine-cosine modes, followed by Nmodes lines for cosine-sine modes, followed by Nmodes lines for cosine-cosine modes.  For the cylinder geometry, there are Nmodes lines for Bessel-sine modes, followed Nmodes lines for Bessel-cosine modes.  If the file filebaseic.dat exists, the script will read a line of text which contains the surface potentials values and height, separated by spaces followed by a line containing the coordinates of the surface mesh points.  

# Output file formats
For faraday.py script, the script will always append to (or create) the file filebase.txt file a line of text containing the applied frequency, acceleration, measured frequency, measured growth rate, measured wave number, substrate inhomogeneity height, and random seed.  If the output flag is set to 1, the script will list more information in filebase.txt in a readable format, a filebasefs.dat containing the final surface height, which can be used as an initial condition in other simulations by copying it to a filebaseic.dat, and also create binary files filebase.npy containing the mesh evolution in binary format and filebasenorms.npy containing the surface disturbance norms, which can be parsed with the Mathematica plot notebooks.  

For the accel.py script, the script will append a line to filebase.txt each time data is saved which contains the measurement count, the measured acceleration amplitude, the measured frequency, the measured initial phase, the measured mean z acceleration, the measured mean x acceleration, and the measured mean y acceleration. It will also create files filebase{x,y,z}count.npy for each saved data which contain binary time series for each direction.  Every time the data is plotted, a file filebasecount.pdf is also saved, and when the saved data is plotted, a file filebase.pdf is created.
