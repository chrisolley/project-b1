# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from b1 import nll_2d
from read_data import read_data
from min_2d import grad_min, newton_min
from helper import latexfigure

# latex file figures- comment out this line if just plotting
latexfigure(0.5)

N = 10000 # number of data points
lifetime, uncertainty = read_data(N) # read data from lifetime.txt 

# perform minimisation with newton's method and gradient method
start_point = (0.46, 0.95) # choose starting point near minimum
sol_grad = grad_min(nll_2d, start_point, 10**(-5), lifetime, uncertainty)
sol_newton = newton_min(nll_2d, start_point, lifetime, uncertainty)

# unpack tuple of results
x_hist_grad, f_hist_grad, x_min_grad, f_min_grad = sol_grad
x_hist_newton, f_hist_newton, x_min_newton, f_min_newton = sol_newton

# set up grids for plotting
# 2d nll function grids
grid_points = 10 # time required to run increases with grid_points^2
tau = np.linspace(0.38, 0.46, grid_points) # range for plotting tau
a = np.linspace(0.94, 1., grid_points) # range for plotting a
T, A = np.meshgrid(tau,a) # set up mesh for contour plotting
nll_matrix = np.zeros((grid_points, grid_points)) # initialise array for 2d nll values

# fill 2d nll matrix row by row 
for i, t in enumerate(tau):
    print('\r Filling 2D NLL function array grid row: {}/{}'.format(i + 1, len(tau)), end="")
    for j, b in enumerate(a):
        nll_matrix[j, i] = nll_2d(tau=t, a=b, lifetime=lifetime, uncertainty=uncertainty)

#Plotting
# plot 2d nll as contour with minimisation visualised as overlaid points
fig1, ax1 = plt.subplots()
nll_2d_min, nll_2d_max = nll_matrix.min(), nll_matrix.max() # dynamic range for colorbar
v = np.linspace(nll_2d_min, nll_2d_max, 10, endpoint=True) # set up range for colorbar
contours = ax1.contour(T, A, nll_matrix, 10, colours='black') # contour lines
contourf = ax1.contourf(T, A, nll_matrix, levels=v, cmap=cm.viridis, alpha=0.5) # filled contour

# plot the iterations of both algorithms
ax1.plot([a[0] for a in x_hist_grad], [a[1] for a in x_hist_grad], color='green', 
         linestyle='--', marker='.', label='Gradient Method')
ax1.plot([a[0] for a in x_hist_newton], [a[1] for a in x_hist_newton], color='red', 
         linestyle='--', marker='.', label='Newton\'s Method')

fig1.colorbar(contourf) # colorbar
plt.clabel(contours, inline=True) # value of contour lines
ax1.grid()
ax1.set_ylabel(r"$a$") 
ax1.set_xlabel(r"$\tau \mathrm{(ps)}$")
plt.tight_layout()

# plot convergence of minimisation algorithms
fig2, ax2 = plt.subplots()
ax2.plot(range(len(x_hist_grad)), [a for a in f_hist_grad], 
         color='green', linestyle='-', marker='.', label='Gradient Method')
ax2.plot(range(len(x_hist_newton)), [a for a in f_hist_newton], 
         color='red', linestyle='-', marker='.', label='Newton\'s Method')
ax2.set_xlabel("Steps")
ax2.set_ylabel("NLL")
ax2.grid()
plt.tight_layout()

# plot 3d view of both algorithms
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
# plot 2d nll function as a surface
ax3.plot_surface(T, A, nll_matrix)
# plot history of both algorithms
ax3.plot([a[0] for a in x_hist_grad], [a[1] for a in x_hist_grad], 
         [a for a in f_hist_grad], color='green')    
ax3.plot([a[0] for a in x_hist_newton], [a[1] for a in x_hist_newton], 
         [a for a in f_hist_newton], color='red')
ax3.set_xlabel(r"$\tau \mathrm{(ps)}$") 
ax3.set_ylabel(r"$a$")
ax3.set_zlabel("NLL")
plt.tight_layout()

plt.show()

# compare steps required for both methods
print('Gradient Method number of steps: {}'.format(len(x_hist_grad)))
print('Gradient Method solution: {}'.format(x_min_grad))
print('Newton\'s Method number of steps: {}'.format(len(x_hist_newton)))
print('Newton\'s Method solution: {}'.format(x_min_newton))