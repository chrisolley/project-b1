# -*- coding: utf-8 -*-
from read_data import read_data
from scipy import stats
from min_1d import para_min, sd_1, sd_2
from b1 import nll
import numpy as np
import matplotlib.pyplot as plt
from helper import latexfigure

def sd_analysis(data_points_list):
    #min_val_array = [] # create array to hold minimised values
    sd_array = [] # create array to hold sd values
    
    for i in data_points_list:
        print("\r Reading {} data points".format(i), end="")
        lifetime, uncertainty = read_data(i)
        nll_minimisation = para_min(nll, (0.1, 0.3, 1.0), 10**(-5), lifetime, uncertainty)
        # unpack tuple of minimisation results
        min_val, points_hist, points_initial, points = nll_minimisation
        s_2 = sd_2(points)
        sd_array.append(s_2)
        #min_val_array.append(min_val[0])
        
    return sd_array

def uncertainty(N):
    
    lifetime, uncertainty = read_data(N)
    nll_minimisation = para_min(nll, (0.1, 0.3, 1.0), 10**(-5), lifetime, uncertainty)
    points = nll_minimisation[3]
    min_val = nll_minimisation[0]
    s_lower, s_upper = sd_1(nll, min_val, lifetime, uncertainty)
    s_2 = sd_2(points)
    
    return s_2, s_lower, s_upper

if __name__ == "__main__":
    latexfigure(0.7)
    data_points_list = range(200,10000,100)
    data_points_list_log = [np.log(i) for i in data_points_list]
    #sd_array = sd_analysis(data_points_list)
    sd_1_lower_array = []
    sd_1_upper_array = []
    sd_2_array = []
    
    for i, a in enumerate(data_points_list):
        print("\r Reading {} data points".format(a), end="")
        sol = uncertainty(a)
        sd_2_array.append(np.log(sol[0]))
        sd_1_lower_array.append(np.log(sol[1]))
        sd_1_upper_array.append(np.log(sol[2]))
        
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(data_points_list_log, sd_2_array)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(data_points_list_log, sd_1_lower_array)
    slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(data_points_list_log, sd_1_upper_array)
    
    print("Slopes: {}, {}, {}.".format(slope1, slope2, slope3))
    print("Intercepts: {}, {}, {}.".format(intercept1, intercept2, intercept3))
    
    fig, ax = plt.subplots()
    ax.plot(data_points_list_log, sd_2_array, color='green', 
            linestyle='', marker='+', markersize=4)
    ax.plot(data_points_list_log, sd_1_lower_array, color='red', 
            linestyle='', marker='+', markersize=4)
    ax.plot(data_points_list_log, sd_1_upper_array, color='red', 
            linestyle='', marker='+', markersize=4)

    ax.plot(data_points_list_log, [-0.5 * i for i in data_points_list_log],
            linestyle='--', color = "blue")
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.grid()
    ax.set_ylabel(r"$\log{(\sigma)}$")
    ax.set_xlabel(r"$\log{(N)}$")
    plt.show()