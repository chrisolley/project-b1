# -*- coding: utf-8 -*-
# !python3
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def para_min(f, x, *args):
    '''
    para_min: Computes the minimum of the parabola formed by three points.
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


def minimise(f, start_points, epsilon, *args):
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
    points_initial = zip(start_points, [f(p, *args) for p in start_points])
    points = zip(start_points, [f(p, *args) for p in start_points])
    points_hist = []  # initialise array to hold past points

    for i in range(2):  # runs the algorithm twice
        x_update = para_min(f, [elem[0] for elem in points], *args)
        point_update = (x_update, f(x_update, *args))
        points.append(point_update)
        points_hist.append(point_update)
        print points
        points.sort(key=lambda tup: tup[1])
        print points
        points.pop()

    conv = abs((points_hist[-1][0] -
                points_hist[-2][0]) / points_hist[-2][0])

    i = 1
    while (conv > epsilon):
        print 'Parabolic minimisation loop {}'.format(i)
        x_update = para_min(f, [elem[0] for elem in points], *args)
        point_update = (x_update, f(x_update, *args))
        points.append(point_update)
        points_hist.append(point_update)
        print points
        points.sort(key=lambda tup: tup[1])
        print points
        points.pop()
        conv = abs((points_hist[-1][0] -
                    points_hist[-2][0]) / points_hist[-2][0])
        i += 1

    return points_hist, points_initial


def f(x, a, b):

    return a * np.cosh(x) + b

# points_hist = minimise(f,(-5, 7, 10),0.01,2,2)
# print [elem[0] for elem in points_hist]
# print [elem[1] for elem in points_hist]
# x = np.linspace(-5,5,100)
# plt.plot(x, f(x, 2, 2), color='blue', linestyle='-')
# plt.plot([elem[0] for elem in points_hist], 
# [elem[1] for elem in points_hist], color='red', linestyle='', marker='.')
# plt.grid()
# plt.show()