# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from b1 import nll_2d

f = open('lifetime.txt')  # measured lifetimes and uncertainty data
lifetime = []
uncertainty = []

for line in f:  # reading data line by line
    data = line.split()
    lifetime.append(float(data[0]))
    uncertainty.append(float(data[1]))

N = len(lifetime)  # number of data points
grid_points = 20
tau = np.linspace(10**(-5), 5., grid_points) # range for nll function
a = np.linspace(10**(-5), 1., grid_points)
nll_matrix = np.zeros((grid_points, grid_points)) # initialise array for nll function plotting

for i, t in enumerate(tqdm(tau, desc="Tau loop")):# tau loop {}'.format(i)):
    for j, b in enumerate(tqdm(a, desc="a loop", leave=False)):# a loop {}'.format(j)):
        #tqdm(tau, total=len(tau), desc='NLL array filling...'):
        nll_matrix[j, i] = nll_2d(tau=t, a=b, lifetime=lifetime, uncertainty=uncertainty)
        

if __name__ == "__main__":
    
    fig1, ax1 = plt.subplots()
    T, A = np.meshgrid(tau,a)
    fig1, ax1 = plt.subplots()
    contour = ax1.contour(T, A, nll_matrix)
    contour = ax1.contourf(T, A, nll_matrix, cmap=cm.viridis)
    fig1.colorbar(contour)
    ax1.grid()
    ax1.set_ylabel("a")
    ax1.set_xlabel("tau")
    plt.show()