# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
import math

def fit(t,tau,s): 

	'''
	fit: Theoretical distribution for the measurements of decay lifetimes of subatomic particles
	Args: 
		t: measured lifetime (independent variable)
		tau: average lifetime
		s: measurement error
	'''

	return (1./(2.*tau))*np.exp((s**2/(2*tau**2))-t/tau)*math.erfc((1./np.sqrt(2))*(s/tau-t/s))


def nll(lifetime, uncertainty, tau):
	'''
	nll: Negative Log Likelihood, calculates the likelihood of an ensemble of PDFs.
	Args:
		tau: average lifetime.
	Returns: 
		nll: negative log likelihood.
	'''
	prob_list = []

	for (t,u) in zip(lifetime, uncertainty):
		#prob = (1./(2*tau))*np.exp((u**2/(2*tau**2))-(t/tau))*math.erfc((1./np.sqrt(2))*(u/tau-t/u)) # calculate probability of each data point based on theoretical distribution
		prob = fit(t,tau,u)
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


t = np.linspace(0,10,1000)
g = []
tau = np.linspace(0,5,100)
nll_list = []

for a in t: 
        g.append(fit(a,tau=1.,s=.05))

for a in tau: 
	nll_list.append(nll(lifetime, uncertainty, a))

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
ax3.grid()


plt.show()
