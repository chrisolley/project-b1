# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from b1 import nll_2d
from read_data import read_data
from min_2d import grad_min, newton_min, sd
from helper import latexfigure

# latex file figures
latexfigure(0.7)

N = 10000 # number of data points
lifetime, uncertainty = read_data(N) # read data from lifetime.txt 

# perform minimisation with newton's method and gradient method
start_point = (.46, 0.95) # initial guess
sol_grad = grad_min(nll_2d, start_point, 10**(-5), lifetime, uncertainty)
sol_newton = newton_min(nll_2d, start_point, lifetime, uncertainty)

# unpack tuple of results
x_hist_grad, f_hist_grad, x_min_grad, f_min_grad = sol_grad
x_hist_newton, f_hist_newton, x_min_newton, f_min_newton = sol_newton

# set up grids for plotting

# 2d nll function grids
# set up a range, close to minimum around guess for sd range
a_range = np.linspace(x_min_newton[1] - 0.01, x_min_newton[1] + 0.01, 1000)
sd_contour1 = [] # initialise arrays for holding contour values (lower sd)
sd_contour2 = [] # (upper sd)

# iterates over slices of constant a to find the s.d. values for a 1d slice
for i, b in enumerate(a_range):
    print('\r 2D S.D. grid row: {}/{}'.format(i + 1, len(a_range)), end="")
    # calculates the lower sd and appends to list
    # shifts guess away from minimum towards the lower sd 
    sd_contour1.append((sd(nll_2d, f_min_newton, x_min_newton[0]-0.01, 
                           b, lifetime, uncertainty), b))
    # calculates the upper sd and appends to list
    # shifts guess away from minimum towards the upper sd 
    sd_contour2.append((sd(nll_2d, f_min_newton, x_min_newton[0]+0.01,  
                           b, lifetime, uncertainty), b))
    
# initialise array for cleaning contour points
sd_contour = []

# removes all nan values resulting from attempting to solve non solvable equations for sd
for i in sd_contour1:
    if (np.isnan(i[0]) == False): 
        sd_contour.append((i[0], i[1]))

for i in sd_contour2:
    if (np.isnan(i[0]) == False): 
        sd_contour.append((i[0], i[1]))

# calculates the upper and lower sd points for tau and a
tau_sd_upper = max([a[0] for a in sd_contour])
tau_sd_lower = min([a[0] for a in sd_contour])
a_sd_upper = max([a[1] for a in sd_contour])
a_sd_lower = min([a[1] for a in sd_contour])

print('tau uncertainty = + {} - {}'.format(abs(tau_sd_upper - x_min_newton[0]), 
                                           abs(tau_sd_lower - x_min_newton[0])))
print('a uncertainty = + {} - {}'.format(abs(a_sd_upper- x_min_newton[1]), 
                                         abs(a_sd_lower - x_min_newton[1])))

print((tau_sd_lower, a_sd_lower), (tau_sd_lower, a_sd_upper))
print((tau_sd_lower, a_sd_lower), (tau_sd_upper, a_sd_lower))
print((tau_sd_upper, a_sd_lower), (tau_sd_upper, a_sd_upper))
print((tau_sd_lower, a_sd_upper), (tau_sd_upper, a_sd_upper))

# plotting
fig, ax = plt.subplots()
# plots the nll_min+1/2 contour 
ax.plot([a[0] for a in sd_contour], [b[1] for b in sd_contour], 
        color='blue', linestyle='', marker='.')
# plot min value for 2d nll
ax.plot(x_min_newton[0], x_min_newton[1], color='red', linestyle=' ', marker='x')
# plot lines showing upper and lower sd for tau and a
ax.plot([tau_sd_lower, tau_sd_lower], [a_sd_lower, a_sd_upper], color='blue',
        linestyle='--')
ax.plot([tau_sd_lower, tau_sd_upper], [a_sd_lower, a_sd_lower], color='blue',
        linestyle='--')
ax.plot([tau_sd_upper, tau_sd_upper], [a_sd_lower, a_sd_upper], color='blue',
        linestyle='--')
ax.plot([tau_sd_lower, tau_sd_upper], [a_sd_upper, a_sd_upper], color='blue',
        linestyle='--')
ax.grid()
ax.set_ylabel(r"$a$") 
ax.set_xlabel(r"$\tau$")
plt.tight_layout()
plt.show()