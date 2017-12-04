# -*- coding: utf-8 -*-
# !python3
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def min(f, x, *args):
    '''
    min: Computes the minimum of the parabola formed by three points.
    Args:
        f: function giving y coordinate of points.
        x: list of three points.
        *args: function arguments.

    Returns:
        x_min: x coord of minimum.
    '''

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
    f = data[0][1] * ((x - data[1][0]) * (x - data[2][0])) / ((data[0][0] - data[1][0]) * (data[0][0] - data[2][0])) + data[1][1] * ((x - data[0][0]) * (x - data[2][0])) / ((data[1][0] - data[0][0]) * (data[1][0] - data[2][0])) + data[2][1] * ((x - data[0][0]) * (x - data[1][0])) / ((data[2][0] - data[0][0]) * (data[2][0] - data[1][0]))

    return f


def sd_1(f, min_val, *args):
    '''
    sd_1: Computes the standard deviation based on a given NLL function and its
        minimum, by varying the NLL function by 1/2.
    Args:
        f: NLL function.
        min_val: Minimum.
    Returns:
        (s_upper, s_lower)
    '''
    # choose a step size relative to the\scale of the function around the min
    step_size = 10**(-3) * min_val[0]

    # calculate upper sd
    print("\n")
    n = 1
    diff = f(min_val[0] + n * step_size, *args) - min_val[1]

    while (diff < 0.5):
        print("\r Upper sd loop {}".format(n), end="")
        n += 1
        diff = f(min_val[0] + n * step_size, *args) - min_val[1]

    s_upper = n * step_size

    # calculate lower sd
    print("\n")
    n = 1
    diff = f(min_val[0] - n * step_size, *args) - min_val[1]

    while (diff < 0.5): 
        print("\r Lower sd loop {}".format(n), end="")
        n += 1
        diff = f(min_val[0] - n * step_size, *args) - min_val[1]

    s_lower = n * step_size
    print ("\n")

    return s_lower, s_upper


def sd_2(p):

    '''
    sd_2: Computes the standard deviation based on the curvature of the
    parabolic approximation of a NLL function. 
    '''
	
    curv = 2 * ((p[0][1]) / ((p[0][0] - p[1][0]) * (p[0][0] - p[2][0])) +
					 (p[1][1]) / ((p[1][0] - p[0][0]) * (p[1][0] - p[2][0])) +
					 (p[2][1]) / ((p[2][0] - p[0][0]) * (p[2][0] - p[1][0])))
    
    s = np.sqrt(1 / curv)

    return s


def para_min(f, start_points, epsilon, *args):
    '''
    minimise: Minimises a function using parabolic minimisation algorithm.
    Args:
        f: function to be minimised.
        start_points: list of three initial points ,must be different.
        epsilon: required precision.
        *args: function arguments.
    '''
    # print f(start_points, *args)
    # creates a tuple of points based on the starting x values
    points_initial = list(zip(start_points, [f(p, *args) for p in start_points]))
    points = list(zip(start_points, [f(p, *args) for p in start_points]))
    points_hist = []  # initialise array to hold past points

    for i in range(2):  # runs the algorithm twice
        x_update = min(f, [elem[0] for elem in points], *args)
        point_update = (x_update, f(x_update, *args))
        points.append(point_update)
        points_hist.append(point_update)
        points.sort(key=lambda tup: tup[1])
        points.pop()

    conv = abs((points_hist[-1][0] -
                points_hist[-2][0]) / points_hist[-2][0])

    i = 1
    print ("\n")
    while (conv > epsilon):
        print("\r Parabolic minimisation loop {}".format(i), end="") 
        x_update = min(f, [elem[0] for elem in points], *args)
        point_update = (x_update, f(x_update, *args))
        points.append(point_update)
        points_hist.append(point_update)
        points.sort(key=lambda tup: tup[1]) # sorts in ascending order
        points.pop() # removes highest point
        conv = abs((points_hist[-1][0] -
                    points_hist[-2][0]) / points_hist[-2][0])
        i += 1
    
    min_val = points_hist[-1]
    
    s_lower1, s_upper1 = sd_1(f, min_val, *args)
    s_2 = sd_2(points)
    
    return min_val, points_hist, points_initial, s_lower1, s_upper1, s_2, points




def f(x, a, b):

    return a * np.cosh(x) + b


if __name__ == "__main__":

    points = ((-7, 1), (4, 5), (-2, 8))
    x = np.linspace(-10, 10, 100)
    y = [quad_poly(a, points) for a in x]
    plt.plot(points[0][0], points[0][1], color='red', linestyle='', marker='.')
    plt.plot(points[1][0], points[1][1], color='red', linestyle='', marker='.')
    plt.plot(points[2][0], points[2][1], color='red', linestyle='', marker='.')
    plt.plot(x, y)
    plt.show()