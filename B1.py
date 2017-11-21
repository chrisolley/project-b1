# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from minimizer import minimise

def fit(tau, t, s): 

	'''
	fit: Theoretical distribution for the measurements of decay lifetimes of subatomic particles
	Args: 
		t: measured lifetime (independent variable)
		tau: average lifetime
		s: measurement error
	'''

	return (1./(2.*tau))*np.exp((s**2/(2*tau**2))-t/tau)*math.erfc((1./np.sqrt(2))*(s/tau-t/s))


def nll(tau, lifetime, uncertainty):
	'''
	nll: Negative Log Likelihood, calculates the likelihood of an ensemble of PDFs.
	Args:
		tau: average lifetime.
		lifetime: list of lifetime measurements.
		uncertainty: list of uncertainties associated with lifetime measurements.
	Returns: 
		nll: negative log likelihood.
	'''
	prob_list = []

	for (t,s) in zip(lifetime, uncertainty):
		prob = fit(tau,t,s)
		prob_list.append(np.log(prob))

	nll = -sum(prob_list)

	return nll

#Filling arrays & reading data

f = open('lifetime.txt') # measured lifetimes and uncertainty data
lifetime = []
uncertainty = []

for line in f: # reading data
	data = line.split()
	lifetime.append(float(data[0]))
	uncertainty.append(float(data[1]))

N = len(lifetime) # number of data points


t = np.linspace(0.,10.,1000)
g = []
tau = np.linspace(0.,5.,100)
nll_list = []

for a in tqdm(t, total=len(t), desc='Fit function array filling...'): 
        g.append(fit(tau=1.,t=a,s=.05))

for a in tqdm(tau, total=len(tau), desc='NLL array filling...'): 
	nll_list.append(nll(tau=a, lifetime=lifetime, uncertainty=uncertainty))


points_hist = minimise(nll, (0.1, 0.3, 1.0), 10**(-5), lifetime, uncertainty)
#TODO what starting values does this work for, how to treat negative values that appear from the minimisation algorithm.
print(points_hist[-1])

#Plotting

fig1,ax1 = plt.subplots()
ax1.hist(lifetime, bins = int(np.sqrt(N)))
ax1.grid()
ax1.set_xlim(0,7)

fig2,ax2 = plt.subplots()
ax2.plot(t,g)
ax2.set_xlim(0,7)
ax2.grid()

fig3,ax3 = plt.subplots()
ax3.plot(tau, nll_list)
ax3.plot([elem[0] for elem in points_hist], [elem[1] for elem in points_hist], color='red', linestyle='', marker='.')
ax3.grid()

plt.show()
