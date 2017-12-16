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
start_point = (.46, 0.95)
sol_grad = grad_min(nll_2d, start_point, 10**(-5), lifetime, uncertainty)
sol_newton = newton_min(nll_2d, start_point, lifetime, uncertainty)

# unpack tuple of results
x_hist_grad, f_hist_grad, x_min_grad, f_min_grad = sol_grad
x_hist_newton, f_hist_newton, x_min_newton, f_min_newton = sol_newton

# set up grids for plotting

# 2d nll function grids
a_range = np.linspace(x_min_newton[1] - 0.01, x_min_newton[1] + 0.01, 45)
sd_contour1 = []
sd_contour2 = []

for i, b in enumerate(a_range):
    print('\r 2D S.D. grid row: {}/{}'.format(i + 1, len(a_range)), end="")
    sd_contour1.append((sd(nll_2d, f_min_newton, x_min_newton[0]-0.01, 
                           b, lifetime, uncertainty), b))
    sd_contour2.append((sd(nll_2d, f_min_newton, x_min_newton[0]+0.01, 
                           b, lifetime, uncertainty), b))

sd_contour = []

for i in sd_contour1:
    if (np.isnan(i[0]) == False): 
        sd_contour.append((i[0], i[1]))

for i in sd_contour2:
    if (np.isnan(i[0]) == False): 
        sd_contour.append((i[0], i[1]))

tau_sd_upper = abs(max([a[0] for a in sd_contour]) - x_min_newton[0])
tau_sd_lower = abs(min([a[0] for a in sd_contour]) - x_min_newton[0])
a_sd_upper = abs(max([a[1] for a in sd_contour]) - x_min_newton[1])
a_sd_lower = abs(min([a[1] for a in sd_contour]) - x_min_newton[1])

print('tau uncertainty = + {} - {}'.format(tau_sd_upper, tau_sd_lower))
print('a uncertainty = + {} - {}'.format(a_sd_upper, a_sd_lower))

fig, ax = plt.subplots()
ax.plot([a[0] for a in sd_contour], [b[1] for b in sd_contour], 
        color='blue', linestyle=' ', marker='.')
ax.plot(x_min_newton[0], x_min_newton[1], color='red', linestyle=' ', marker='.')
ax.grid()
ax.set_ylabel(r"$a$") 
ax.set_xlabel(r"$\tau$")
plt.tight_layout()
plt.show()