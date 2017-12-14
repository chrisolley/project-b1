     # -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from read_data import read_data
from min_1d import para_min, quad_poly, sd_1, sd_2
from b1 import nll, fit

f = open('lifetime.txt')  # measured lifetimes and uncertainty data
lifetime = []
uncertainty = []

for line in f:  # reading data line by line
    data = line.split()
    lifetime.append(float(data[0]))
    uncertainty.append(float(data[1]))

N = len(lifetime)  # number of data points

t = np.linspace(5*10**(-2), 10., 1000) # range for fit function
g = [] # initialise array for fit function plotting
tau = np.linspace(5*10**(-2), 5., 100) # range for nll function
nll_list = [] # initialise array for nll function plotting
nll_sd_list = [] # initialise array for nll sd plotting

# fills fit function array
for i, a in enumerate(t):
    print('\r Filling fit function array: {}/{}'.format(i, len(t)), end="")
    g.append(fit(tau=.5, t=a, s=.1))
print("\n")
# fills nll function array
for i, a in enumerate(tau):
    print('\r Filling NLL array: {}/{}'.format(i, len(tau)), end="")
    nll_list.append(nll(tau=a, lifetime=lifetime, uncertainty=uncertainty))

# minimise the nll function
nll_minimisation = para_min(nll, (0.1, 0.3, 1.0), 10**(-5), lifetime, uncertainty)

# unpack tuple of minimisation results
min_val, points_hist, points_initial, points = nll_minimisation

s_lower1, s_upper1 = sd_1(nll, min_val, lifetime, uncertainty)
s_2 = sd_2(points)

# create range for tau estimate standard deviation plotting
tau_sd = np.linspace(min_val[0] - 2 * s_lower1, min_val[0] + 2 * s_upper1, 100)

# fill nll function array focused on region around standard deviation points
for a in tau_sd:
    print('\r Filling NLL array for S.D. plotting: {}/{}'.format(i, len(tau_sd)), end="")
    nll_sd_list.append(nll(tau=a, lifetime=lifetime, uncertainty=uncertainty))

# display results of minimisation
print("\n")
print("NLL function is minimised (NLL = {}) for: ".format(min_val[1]))
print("tau = {}, ".format(min_val[0]))
print("s.d. = + {} - {} (NLL Variation Method), ".format(s_upper1, s_lower1))
print("s.d. = +/- {} (Curvature Method).".format(s_2))

# plot histogram of data
print("\n")
print("Plotting...")
fig1, ax1 = plt.subplots()
ax1.hist(lifetime, bins=int(np.sqrt(N)), normed=True, edgecolor='black')
# plot example of fit function
ax1.plot(t, g)
ax1.set_xlim(0, 5)
ax1.grid()

# visualise minimisation algorithm for nll function
fig3, ax3 = plt.subplots()
ax3.plot(tau, nll_list)
ax3.plot([elem[0] for elem in points_hist],
         [elem[1] for elem in points_hist],
         color='red', linestyle='', marker='.')

ax3.plot([elem[0] for elem in points_initial],
         [elem[1] for elem in points_initial],
         color='green', linestyle='', marker='.')

ax3.set_ylabel("NLL")
ax3.set_xlabel("tau")
ax3.grid()

# visualise standard deviation algorithm for nll function
fig4, ax4 = plt.subplots()
ax4.plot(tau_sd, nll_sd_list)
ax4.plot(min_val[0], min_val[1], color='red', linestyle='', marker='.', 
         label="Minimum")
ax4.plot([min_val[0]+s_upper1, min_val[0]-s_lower1], 
         [min_val[1]+0.5, min_val[1]+0.5] , color='blue', 
         linestyle='', marker='.', label='S.D.')
ax4.plot([elem[0] for elem in points],
         [elem[1] for elem in points],
         color='red', linestyle='', marker='.', label='Final points')
y = [quad_poly(a, points) for a in tau_sd]
ax4.plot(tau_sd, y, label='Quadratic fit')

ax4.set_xlim(min_val[0] - 2 * s_lower1, min_val[0] + 2 * s_upper1)
ax4.set_ylim(min_val[1] - 1.5, min_val[1] + 1.5)
ax4.set_ylabel("NLL")
ax4.set_xlabel("tau")
ax4.set_title("S.D. estimated by method 1")
ax4.legend(loc="best")
ax4.grid()

plt.show()