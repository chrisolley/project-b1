# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from b1 import nll_2d
from min_2d import grad_min, newton_min

f = open('lifetime.txt')  # measured lifetimes and uncertainty data
lifetime = []
uncertainty = []

for line in f:  # reading data line by line
    data = line.split()
    lifetime.append(float(data[0]))
    uncertainty.append(float(data[1]))

N = len(lifetime)  # number of data points

#start_point = (.4, .6) #(0.4, 0.6) works
start_point = (.4, .8)
#sol1 = grad_min(nll_2d, start_point, 0.2, lifetime, uncertainty)
sol2 = newton_min(nll_2d, start_point, lifetime, uncertainty)
#x_hist1 = sol1[0]
#print(sol1[0])
x_hist2 = sol2[0]
f_hist2 = sol2[1]
print(sol2[0])

grid_points = 20
tau = np.linspace(0.2, 1., grid_points) # range for nll function
a = np.linspace(0.6, 1., grid_points)

nll_matrix = np.zeros((grid_points, grid_points)) # initialise array for nll function plotting

for i, t in enumerate(tqdm(tau, desc="Tau loop")):# tau loop {}'.format(i)):
    for j, b in enumerate(tqdm(a, desc="a loop", leave=False)):# a loop {}'.format(j)):
    #tqdm(tau, total=len(tau), desc='NLL array filling...'):
        nll_matrix[j, i] = nll_2d(tau=t, a=b, lifetime=lifetime, uncertainty=uncertainty)

#Plotting
fig1, ax1 = plt.subplots()
T, A = np.meshgrid(tau,a)

nll_2d_min, nll_2d_max = nll_matrix.min(), nll_matrix.max()

contours = ax1.contour(T, A, nll_matrix, 10, colours='black')
plt.clabel(contours, inline=True)

v = np.linspace(nll_2d_min, nll_2d_max, 10, endpoint=True)
contourf = ax1.contourf(T, A, nll_matrix, levels=v, cmap=cm.viridis, alpha=0.5)
#contour = ax1.contour(T, A, nll_matrix)
#ax1.plot([a[0] for a in x_hist1], [a[1] for a in x_hist1], color='green', 
#         linestyle='--', marker='.', label='Gradient Method')
#ax1.imshow(nll_matrix, extent=[tau.min(),tau.max(), a.min(), a.max()],
#           cmap='RdGy', alpha=.5)

ax1.plot([a[0] for a in x_hist2], [a[1] for a in x_hist2], color='red', 
         linestyle='--', marker='.', label='Newton\'s Method')

fig1.colorbar(contourf)
ax1.grid()
ax1.set_ylabel("a")
ax1.set_xlabel("tau")
ax1.legend(loc='best')

fig2, ax2 = plt.subplots()
#ax2.plot(range(len(x_hist1)), [a[0] for a in x_hist1], color='green', linestyle='-', marker='.', label='Gradient Method')
ax2.plot(range(len(x_hist2)), [a for a in f_hist2], color='red', linestyle='-', marker='.', label='Newton\'s Method')
ax2.grid()
ax2.legend(loc='best')

plt.show()

#print('Gradient Method number of steps: {}'.format(len(x_hist1)))
print('Newton\'s Method number of steps: {}'.format(len(x_hist2)))