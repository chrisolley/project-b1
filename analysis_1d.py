     # -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from min_1d import para_min, quad_poly
from b1 import nll, fit

f = open('lifetime.txt')  # measured lifetimes and uncertainty data
lifetime = []
uncertainty = []

for line in f:  # reading data line by line
    data = line.split()
    lifetime.append(float(data[0]))
    uncertainty.append(float(data[1]))

N = len(lifetime)  # number of data points

t = np.linspace(10**(-5), 10., 1000) # range for fit function
g = [] # initialise array for fit function plotting
tau = np.linspace(10**(-5), 5., 100) # range for nll function
nll_list = [] # initialise array for nll function plotting
nll_sd_list = [] # initialise array for nll sd plotting

# fills fit function array
for a in tqdm(t, total=len(t), desc='Fit function array filling...'):
    g.append(fit(tau=1., t=a, s=.05))

# fills nll function array
for a in tqdm(tau, total=len(tau), desc='NLL array filling...'):
    nll_list.append(nll(tau=a, lifetime=lifetime, uncertainty=uncertainty))

# minimise the nll function
nll_minimisation = para_min(nll, (0.1, 0.3, 1.0), 10**(-5), lifetime, uncertainty)

# unpack tuple of minimisation results
min_val, points_hist, points_initial, s_lower1, s_upper1, s_2, points = nll_minimisation

# create range for tau estimate standard deviation plotting
tau_sd = np.linspace(min_val[0] - 2 * s_lower1, min_val[0] + 2 * s_upper1, 100)

# fill nll function array focused on region around standard deviation points
for a in tqdm(tau_sd, total=len(tau_sd), desc='NLL array filling to display S.D...'):
    nll_sd_list.append(nll(tau=a, lifetime=lifetime, uncertainty=uncertainty))

# display results of minimisation
print("NLL function is minimised (NLL = {}) for a value of tau = \
      {}, s.d. + {} - {} (Method 1), sd = +/- {} (Method 2)".format(
      min_val[1], min_val[0], s_upper1, s_lower1, s_2))

# plot histogram of data
fig1, ax1 = plt.subplots()
ax1.hist(lifetime, bins=int(np.sqrt(N)))
ax1.set_xlim(0, 7)
ax1.grid()

# plot example of fit function
fig2, ax2 = plt.subplots()
ax2.plot(t, g)
ax2.set_xlim(0, 7)
ax2.grid()

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