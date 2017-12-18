# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from helper import newton_raphson


def minimum(f, x, *args):
    '''
    minimum: Computes the minimum of the parabola formed by three points.
    Args:
        f: function giving y coordinate of points.
        x: list of three points.
        *args: function arguments.

    Returns:
        x_min: x coord of minimum.
    '''
    
    # calculate the analytic expression for the minimum of a parabola
    
    num = 0.5 * ((x[2]**2 - x[1]**2) * f(x[0], *args) +
                 (x[0]**2 - x[2]**2) * f(x[1], *args) +
                 (x[1]**2 - x[0]**2) * f(x[2], *args))
    denom = ((x[2] - x[1]) * f(x[0], *args) +
             (x[0] - x[2]) * f(x[1], *args) +
             (x[1] - x[0]) * f(x[2], *args))
    
    x_min = num / denom
    
    return x_min


def quad_poly(x, data):
    '''
    quad_poly: Computes the 2nd order lagrange polynomial
               based on 3 data points.
    Args:
        x: evaluation point for the polynomial.
        data: tuple of 3 data points in (x,y) form.

    Returns:
        f: lagrange polynomial evaluated at x.
    '''
    # construct lagrange parabola using analytic form
    
    f = data[0][1] * ((x - data[1][0]) * (x - data[2][0])) / ((data[0][0] - data[1][0]) * (data[0][0] - data[2][0])) + data[1][1] * ((x - data[0][0]) * (x - data[2][0])) / ((data[1][0] - data[0][0]) * (data[1][0] - data[2][0])) + data[2][1] * ((x - data[0][0]) * (x - data[1][0])) / ((data[2][0] - data[0][0]) * (data[2][0] - data[1][0]))

    return f


def sd_1(f, min_val, *args):
    '''
    sd_1: Computes either the upper or lower standard deviation based on a given 
    NLL function and its minimum, by varying the NLL function by 1/2 and solving using newton-raphson.
    Args:
        f: NLL function.
        min_val: Minimum coordinate of function.
    Returns:
        (s_upper, s_lower): upper and lower standard deviation in tuple form.
    '''
    
    epsilon = 10**(-4) # desired precision of n-r solution
    
    # construct function to be solved for the sd values.
    def g(x, *args): 
        return f(x, *args) - (min_val[1] + 0.5)
    
    # solve equation for sd values using newton raphson, using initial guess that
    # shifts away from the minimum towards either the upper or lower sd value.
    s_upper = abs(newton_raphson(g, min_val[0]+0.01, epsilon, *args) - min_val[0])
    s_lower = abs(newton_raphson(g, min_val[0]-0.01, epsilon, *args) - min_val[0])
    
    return (s_upper, s_lower)


def sd_2(p):

    '''
    sd_2: Computes the standard deviation based on the curvature of the
    parabolic approximation of a 2d NLL function. 
    Args:
        p: 3 points approximating the minimum of the NLL function in (3,2)-tuple form.
    Returns: 
        s: standard deviation corresponding to the curvature based on these points.
    '''
    
	# calculate analytic form of curvature for a parabola
    curv = 2 * ((p[0][1]) / ((p[0][0] - p[1][0]) * (p[0][0] - p[2][0])) +
					 (p[1][1]) / ((p[1][0] - p[0][0]) * (p[1][0] - p[2][0])) +
					 (p[2][1]) / ((p[2][0] - p[0][0]) * (p[2][0] - p[1][0])))
    
    # uses lower bound of cramer-rao inequality for sd
    s = np.sqrt(1 / curv)

    return s


def para_min(f, start_points, epsilon, *args):
    '''
    minimise: Minimises a function using parabolic minimisation algorithm.
    Args:
        f: function to be minimised.
        start_points: list of three initial points, must be different and in a
        region of positive curvature.
        epsilon: required precision.
        *args: function arguments.
    Returns: 
        min_val: x value of minimum. 
        points_hist: iteration history as a list of (x,y) tuples.
        points_initial: initial points as a (xstart, ystart) tuple.
        points: final 3 points used in the algorithm in (x,y) tuple form.
    '''
    
    # creates a tuple of points based on the starting x values
    points_initial = list(zip(start_points, [f(p, *args) for p in start_points]))
    points = list(zip(start_points, [f(p, *args) for p in start_points]))
    points_hist = []  # initialise array to hold past points

    for i in range(2):  # runs the algorithm twice to be able to evaluate convergence criteria
        x_update = minimum(f, [elem[0] for elem in points], *args)
        point_update = (x_update, f(x_update, *args))
        points.append(point_update)
        points_hist.append(point_update)
        # sort current points list in ascending order based on y value
        points.sort(key=lambda tup: tup[1])
        points.pop() # remove highest value from list

    # evaluate convergence criteria
    conv = abs((points_hist[-1][0] -
                points_hist[-2][0]) / points_hist[-2][0])

    i = 1
    print ("\n")
    # run algorithm until convergence
    while (conv > epsilon):
        print("\r Parabolic minimisation loop: {}".format(i), end="") 
        
        x_update = minimum(f, [elem[0] for elem in points], *args)
        point_update = (x_update, f(x_update, *args))
        points.append(point_update)
        points_hist.append(point_update)
        points.sort(key=lambda tup: tup[1]) # sorts in ascending order
        points.pop() # removes highest point
        # evaluate convergence criteria
        conv = abs((points_hist[-1][0] -
                    points_hist[-2][0]) / points_hist[-2][0])
        i += 1
    
    min_val = points_hist[-1]
    
    return min_val, points_hist, points_initial, points




def f(x, a, b):
    # test function for parabolic minimisation
    return a * np.cosh(x) + b


if __name__ == "__main__":
    # test quad_poly function for creating parabola.
    points = ((-7, 1), (4, 5), (-2, 8))
    x = np.linspace(-10, 10, 100)
    y = [quad_poly(a, points) for a in x]
    fig1, ax1 = plt.subplots()
    ax1.plot(points[0][0], points[0][1], color='red', linestyle='', marker='.')
    ax1.plot(points[1][0], points[1][1], color='red', linestyle='', marker='.')
    ax1.plot(points[2][0], points[2][1], color='red', linestyle='', marker='.')
    ax1.plot(x, y)
    ax1.grid()
    
    # test parabolic minimisation 
    minimisation_test = para_min(f, (-0.4, 0.3, 0.5), 10**(-5), 2, 1)
    min_val, points_hist, points_initial, points = minimisation_test
    x = np.linspace(-1., 1., 1000)
    fig2, ax2 = plt.subplots()
    # plot test function 
    ax2.plot(x, [f(a, 2, 1) for a in x])
    # plot point history
    ax2.plot([elem[0] for elem in points_hist],
             [elem[1] for elem in points_hist],
             color='red', linestyle='', marker='.')
    # plot original points
    ax2.plot([elem[0] for elem in points_initial],
             [elem[1] for elem in points_initial],
             color='green', linestyle='', marker='.')
    ax2.grid()
    plt.show() 
    print(points_hist)