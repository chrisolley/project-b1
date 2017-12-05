# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from b1 import nll_2d
from min_2d import newton_min

f = open('lifetime.txt')  # measured lifetimes and uncertainty data
lifetime = []
uncertainty = []

for line in f:  # reading data line by line
    data = line.split()
    lifetime.append(float(data[0]))
    uncertainty.append(float(data[1]))

N = len(lifetime)  # number of data points
grid_points = 50
tau = np.linspace(0.2, 1., grid_points) # range for nll function
a = np.linspace(0.6, 1., grid_points)
nll_matrix = np.zeros((grid_points, grid_points)) # initialise array for nll function plotting

# for i, t in enumerate(tqdm(tau, desc="Tau loop")):# tau loop {}'.format(i)):
    # for j, b in enumerate(tqdm(a, desc="a loop", leave=False)):# a loop {}'.format(j)):
        # #tqdm(tau, total=len(tau), desc='NLL array filling...'):
        # nll_matrix[j, i] = nll_2d(tau=t, a=b, lifetime=lifetime, uncertainty=uncertainty)
        



# fig1, ax1 = plt.subplots()
# T, A = np.meshgrid(tau,a)

# nll_2d_min, nll_2d_max = nll_matrix.min(), nll_matrix.max()

# v = np.linspace(nll_2d_min, nll_2d_max, 50, endpoint=True)
# contour = ax1.contourf(T, A, nll_matrix, v, cmap=cm.viridis)

# fig1.colorbar(contour)
# ax1.grid()
# ax1.set_ylabel("a")
# ax1.set_xlabel("tau")

sol2 = newton_min(nll_2d, (0.4, 0.8), lifetime, uncertainty)
print(sol2[0])

plt.show()