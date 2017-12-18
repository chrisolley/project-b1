# -*- coding: utf-8 -*-

import numpy as np
import math
 
def fit(t, tau, s):
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

def fit_back(t, s):
    '''
    fit_back: Gaussian distribution of background signal, to be used in constructing the
    2d nll function.
    Args: 
        t: measured lifetime produced by background signal.
    Returns: 
        f: value of distribution at point t.
    '''
    
    f = (s * np.sqrt(2 * np.pi))**(-1) * np.exp(-.5 * (t**2 / s**2))
       
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
        prob = fit(t, tau, s)
        prob_list.append(np.log(prob))

    nll = -sum(prob_list)

    return nll

def nll_2d(tau, a, lifetime, uncertainty):
    '''
    nll_2d: 2D Negative Log Likelihood, takes into account certain proportion
    of background signal given by a. 
    Args: 
        tau: average lifetime.
        a: proportion of true signal in total signal.
        lifetime: list of lifetime measurements. 
        uncertainty: list of uncertainties associated with lifetime measurements.
    Returns: 
        nll: 2d negative log likelihood for a given data set and specified 
        parameters.
    '''
    
    prob_list = []
    for (t,s) in zip(lifetime, uncertainty):
        prob = a * fit(t, tau, s) + (1 - a) * fit_back(t, s)            
        prob_list.append(np.log(prob))
        
    nll = -sum(prob_list)
    
    return nll

