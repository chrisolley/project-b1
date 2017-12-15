     # -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from read_data import read_data
from min_1d import para_min, quad_poly, sd_1, sd_2
from b1 import nll, fit
from helper import latexfigure
 

# latex file figures- comment out this line and above if just plotting
latexfigure(0.7)

N = 10000 # number of data points
lifetime, uncertainty = read_data(N) # read data from lifetime.txt 

#initial data analysis
av_lifetime = sum(lifetime)/N
av_uncertainty = sum(uncertainty)/N
print("Average lifetime from data: {}".format(av_lifetime))
print("Average uncertainty from data: {}".format(av_uncertainty))

t = np.linspace(-3., 3., 1000) # range for fit function
tau = np.linspace(5*10**(-2), 5., 100) # range for nll function

nll_sd_list = [] # initialise array for nll sd plotting

# minimise the nll function
nll_minimisation = para_min(nll, (0.1, 0.3, 1.0), 10**(-5), lifetime, uncertainty)

# unpack tuple of minimisation results
min_val, points_hist, points_initial, points = nll_minimisation

#s_lower1, s_upper1 = sd_1(nll, min_val, lifetime, uncertainty)
#s_lower1 = sd_1(nll, min_val[1], min_val[0] - 0.01, lifetime, uncertainty)
#s_upper1 = sd_1(nll, min_val[1], min_val[0] + 0.01, lifetime, uncertainty)
s_lower1, s_upper1 = sd_1(nll, min_val, lifetime, uncertainty)
s_2 = sd_2(points)

# create range for tau estimate standard deviation plotting
tau_sd = np.linspace(min_val[0] - s_lower1 - 0.005, min_val[0] + s_upper1 + 0.005, 100)

# display results of minimisation
print("\n")
print("NLL function is minimised (NLL = {}) for: ".format(min_val[1]))
print("tau = {}, ".format(min_val[0]))
print("s.d. = + {} - {} (NLL Variation Method), ".format(s_upper1, s_lower1))
print("s.d. = +/- {} (Curvature Method).".format(s_2))

# plot histogram of data overlaid with fit functions with different parameters
print("Plotting...")

fig, ax = plt.subplots()
ax.hist(lifetime, bins=int(np.sqrt(N)), edgecolor='black')
ax.set_xlim(-3., 3.)
ax.set_xlabel(r"$t \mathrm{(ps)}$")
ax.set_ylabel("Frequency")
ax.grid()
plt.tight_layout()

# plotting different values of tau
fig1, ax1 = plt.subplots()
ax1.hist(lifetime, bins=int(np.sqrt(N)), normed=True, edgecolor='black')
ax1.plot(t, [fit(0.4, a, 0.3) for a in t]) # tau = 0.4
ax1.plot(t, [fit(0.5, a, 0.3) for a in t]) # tau = 0.5
ax1.plot(t, [fit(0.3, a, 0.3) for a in t]) # tau = 0.3
ax1.plot(t, [fit(0.2, a, 0.3) for a in t]) # tau = 0.2
ax1.set_xlim(-3., 3.)
ax1.set_xlabel(r"$t \mathrm{(ps)}$")
ax1.set_ylabel(r"$f_{sig}^m$")  
ax1.grid()
plt.tight_layout()

# plotting different values of sigma
fig2, ax2 = plt.subplots()
ax2.hist(lifetime, bins=int(np.sqrt(N)), normed=True, edgecolor='black')
ax2.plot(t, [fit(0.4, a, 0.3) for a in t]) # sigma = 0.3
ax2.plot(t, [fit(0.4, a, 0.2) for a in t]) # sigma = 0.2
ax2.plot(t, [fit(0.4, a, 0.4) for a in t]) # sigma = 0.4
ax2.plot(t, [fit(0.4, a, 0.1) for a in t]) # sigma = 0.1
ax2.set_xlim(-3., 3.)
ax2.set_xlabel(r"$t \mathrm{(ps)}$")
ax2.set_ylabel(r"$f_{sig}^m$")
ax2.grid()
plt.tight_layout()

# visualise minimisation algorithm for nll function
fig3, ax3 = plt.subplots()
ax3.plot(tau, [nll(a, lifetime, uncertainty) for a in tau])
ax3.plot([elem[0] for elem in points_hist],
         [elem[1] for elem in points_hist],
         color='red', linestyle='', marker='.')
ax3.plot([elem[0] for elem in points_initial],
         [elem[1] for elem in points_initial],
         color='green', linestyle='', marker='.')
ax3.set_ylabel("NLL")
ax3.set_xlabel(r"$\tau \mathrm{(ps)}$")
ax3.grid()
plt.tight_layout()

# visualise standard deviation algorithm for nll function
fig4, ax4 = plt.subplots()
ax4.plot(tau_sd, [nll(a, lifetime, uncertainty) for a in tau_sd], color='black')
ax4.plot(min_val[0], min_val[1], color='red', linestyle='', marker='o', markeredgecolor='black')
ax4.plot([min_val[0] + s_upper1, min_val[0] - s_lower1], 
         [min_val[1] + 0.5, min_val[1] + 0.5] , color='blue', 
         linestyle='', marker='o', markeredgecolor='black')
ax4.plot([elem[0] for elem in points],
         [elem[1] for elem in points],
         color='red', linestyle='', marker='o', markeredgecolor='black')
y = [quad_poly(a, points) for a in tau_sd]
ax4.plot(tau_sd, y, linestyle='--')

ax4.set_xlim(min_val[0] - s_lower1 - 0.005 , min_val[0] + s_upper1 + 0.005)
ax4.set_ylim(min_val[1] - 0.5, min_val[1] + 1.5)
ax4.set_ylabel("NLL")
ax4.set_xlabel(r"$\tau \mathrm{(ps)}$")
ax4.grid()

plt.tight_layout()
plt.show()