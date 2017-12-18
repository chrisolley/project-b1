project-b1

This code analyses data from the decay of D0 mesons, using negative log likelihood estimation to estimate the lifetime of this subatomic particle. 

Library dependencies: 
matplotlib
numpy
scipy

Contents: 

lifetime.txt: Lifetime measurement data.

b1.py: Contains the main physical functions relating to the project.

helper.py: Contains various 'helper' functions in order to carry out the analysis, e.g. finite difference methods, newton-raphson methods and matrix methods (e.g. LU decomposition, dot product).

read_data.py: Helper function to read a certain number of data points from lifetime.txt.

min_1d.py: Contains the 1 dimensional parabolic minimiser and functions for calculating the standard deviation of a 1d nll function. Running this script carries out a 1D minimisation test for cosh(x).

min_2d.py: Contains the 2 dimensional minimisation algorithms: gradient method and newton's method, as well as a function to calculate the standard deviation of slices of a 2d nll function. Running this script carries out a 2D minimisation test for both algorithms.

analysis_1d.py: Main analysis script for 1d case. Running this script minimises the 1d NLL function and produces relevant plots, printing results to the console.

analysis_2d.py: Main analysis script for the 2d case. Running this script minimises the 2d NLL function and produces relevant plots visualising the minimisation algorithms and prints the minimisation results to the console.

sd_1d_analysis.py: Additional 1D analysis script, examining the dependence of nll estimate error on dataset size. Carries out a linear regression of this relationship using the scipy.stats.linregress module. Running this script produces relevant plots visualising the minimisation algorithms and prints the minimisation results to the console.

sd_2d.py: Additional 2D analysis script, calculating the NLL_min+1/2 contour for the 2D NLL function, in order to calculate the error on the 2d estimates of tau and a. Running this script produces a plot of this contour and prints sd results to the console.


