# -*- coding: utf-8 -*-
from read_data import read_data
from scipy import stats
from min_1d import para_min, sd_1, sd_2
from b1 import nll
import numpy as np
import matplotlib.pyplot as plt
from helper import latexfigure

def uncertainty(N):
    
    '''
    uncertainty: Calculates the sd for a given number of data points, using
    curvature method and nll variation method. 
    Args: 
        N: number of data points used. 
    Returns: 
        s_2: sd using curvature method. 
        s_lower: lower sd using nll variation method. 
        s_upper: upper sd using nll variation method
    '''
    
    lifetime, uncertainty = read_data(N) # read data for required number of data points
    # minimise function
    nll_minimisation = para_min(nll, (0.1, 0.3, 1.0), 10**(-5), lifetime, uncertainty)
    # unpack required tuple values
    points = nll_minimisation[3]
    min_val = nll_minimisation[0]
    # calculate sd using both methods
    s_lower, s_upper = sd_1(nll, min_val, lifetime, uncertainty)
    s_2 = sd_2(points)
    
    return s_2, s_lower, s_upper


if __name__ == "__main__":
    
    # latex file figures - comment out if just plotting
    latexfigure(0.7)
    
    # list of N values (i.e. data set sizes)
    data_points_list = range(200,10000,100)
    # convert to log form
    data_points_list_log = [np.log(i) for i in data_points_list]

    # initialise arrays to hold error values calculated through diff methods
    sd_1_lower_array = []
    sd_1_upper_array = []   
    sd_2_array = []
    
    # populate error value arrays for N values desired
    for i, a in enumerate(data_points_list):
        print("\r Reading {} data points".format(a), end="")
        sol = uncertainty(a)
        sd_2_array.append(np.log(sol[0])) # curvature method
        sd_1_lower_array.append(np.log(sol[1])) # nll variation method lower
        sd_1_upper_array.append(np.log(sol[2])) # and upper
    
    # uses scipy.stats to perform linear regression on log-log plot    
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(data_points_list_log, sd_2_array)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(data_points_list_log, sd_1_lower_array)
    slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(data_points_list_log, sd_1_upper_array)
    
    print("\n")
    print("Slopes: {}, {}, {}.".format(slope1, slope2, slope3))
    print("Intercepts: {}, {}, {}.".format(intercept1, intercept2, intercept3))
    
    # calculate averages of linear regression parameters
    slope_av = sum((slope1, slope2, slope3))/3
    intercept_av = sum((intercept1, intercept2, intercept3))/3
    std_av = sum((std_err1, std_err2, std_err3))/3
    rval_av = sum((r_value1, r_value2, r_value3))/3
    
    print("Average slope: {}.".format(slope_av))
    print("Average intercept: {}.".format(intercept_av))
    print("Average slope std: {}".format(std_av))
    print("Average r-value: {}".format(rval_av))
    print("For sigma=AN^alpha: ")
    print("A= {}".format(np.exp(intercept_av)))
    print("alpha= {} +- {}".format(slope_av, std_av))   
    
    # plotting
    fig, ax = plt.subplots()
    ax.plot(data_points_list_log, sd_2_array, color='green', 
            linestyle='', marker='+', markersize=4)
    ax.plot(data_points_list_log, sd_1_lower_array, color='red', 
            linestyle='', marker='+', markersize=4)
    ax.plot(data_points_list_log, sd_1_upper_array, color='red', 
            linestyle='', marker='+', markersize=4)
    # plot average linear regression trend for all three sd methods
    ax.plot(data_points_list_log, [slope_av * i + intercept_av for i in data_points_list_log],
            linestyle=':', color = "blue")
    ax.grid()
    ax.set_ylabel(r"$\log{(\sigma)}$")
    ax.set_xlabel(r"$\log{(N)}$")
    plt.show()