# -*- coding: utf-8 -*-
# !python3
import numpy as np
import math
 
 
#plt.rc('text', usetex=True)
#plt.rc('font', family='sans serif')
 
def fit(tau, t, s):
    '''
    fit: Theoretical distribution for the measurements of decay lifetimes
    of subatomic particles. Convolution of an exponential distribution
    and a gaussian.
    Args:
        t: measured lifetime (independent variable).
        tau: average lifetime.
        s: measurement error.
    Returns:
        f: value of lifetime distribution at a point t
        for a given mean lifetime and measurement error.
   '''

    f = (1. / (2. * tau)) * np.exp((s**2 / (2 * tau**2)) - t / tau) * \
    math.erfc((1. / np.sqrt(2)) * (s / tau - t / s))
       
    return f


def nll(tau, lifetime, uncertainty):
    '''
    nll: Negative Log Likelihood, calculates the likelihood of an ensemble
    of PDFs.
    Args:
        tau: average lifetime.
        lifetime: list of lifetime measurements.
        uncertainty: list of uncertainties associated with lifetime
        measurements.
    Returns:
        nll: negative log likelihood.
    '''

    prob_list = []

    for (t, s) in zip(lifetime, uncertainty):
        prob = fit(tau, t, s)
        prob_list.append(np.log(prob))

    nll = -sum(prob_list)

    return nll