# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from helper import dot, lu, newton_raphson, convergence, grad_cds, hessian

def grad_min(f, x, a, *args):
    '''
    grad_min: Minimises a 2d function using the gradient method. 
    Args:
        f: function to be minimized. 
        x: starting point in tuple form (x,y). 
        a: step size. 
        *args: function arguments. 
    Returns: 
        x_hist: record of points in tuple form (x,y). 
        x_min: minimum coordinate in tuple form (x,y).
        f_min: minimum value of function.
    '''
    
    x_hist = []
    f_hist = []
    x = np.asarray(x).reshape((2,1)) # converts tuple to array so works with helper functions
    grad = grad_cds(f, x, *args) # calculate gradient
    #print('x: ', x, 'a: ', a, 'grad: ', grad)
    x_update = x - a * grad # calcuate x_(n+1)
    #print('x_update: ', x_update)
    x_hist.append((x[0,0], x[1,0])) # converts back to tuple for more readable form
    #print((x[0,0], x[1,0]))    
    f_hist.append(f(x[0,0], x[1,0], *args))
    x_hist.append((x_update[0,0], x_update[1,0]))
    f_hist.append(f(x_update[0,0], x_update[1,0], *args))
    #print((x_update[0,0], x_update[1,0]))
    conv = convergence(f, x, x_update, *args) # determine convergence criteria
    x = x_update
	
    while (conv > 10**(-6)):
        grad = grad_cds(f, x, *args)
        #print('x: ', x, 'a: ', a, 'grad: ', grad)
        x_update = x - a * grad
        #print('x_update: ', x_update)
        x_hist.append((x_update[0,0], x[1,0])) #convert back to tuple for more readable form
        f_hist.append(f(x_update[0,0], x_update[1,0], *args))
        #print((x_update[0,0], x_update[1,0]))
        conv = convergence(f, x, x_update, *args) # determine convergence criteria
        x = x_update

    x_min = x_hist[-1]
    f_min = f(x_min[0], x_min[1], *args)
	
    return x_hist, f_hist, x_min, f_min

def newton_min(f, x, *args):
    '''
    newton_min: Minimises a 2d function using Newton's method. 
    Args: 
        f: function to be minimized. 
        x: starting point in tuple form (x,y).
        *args: function arguments.
    Returns: 
        x_hist: record of points in tuple form (x,y). 
        x_min: minimum coordinate.
        f_min: minimum value of function.
        
    '''

    x_hist = []
    f_hist = []
    
    x = np.asarray(x).reshape((2,1))
    grad = grad_cds(f, x, *args)
    hess = hessian(f, x, *args)
    inv_hess = lu(hess, np.identity(2))[2]
    x_hist.append((x[0,0], x[1,0]))
    f_hist.append(f(x[0,0], x[1,0], *args))
    d = dot(inv_hess, grad)
    x_update = x - d
    #print(x_update)
    x_hist.append((x_update[0,0], x_update[1,0]))
    f_hist.append(f(x_update[0,0], x_update[1,0], *args))
    conv = convergence(f, x, x_update, *args) # determine convergence criteria
    x = x_update
	
    while (conv > 10**(-6)):
        grad = grad_cds(f, x, *args)
        hess = hessian(f, x, *args)
        inv_hess = lu(hess, np.identity(2))[2]
        d = dot(inv_hess,grad)
        x_update = x - d
        #print(x_update)
        x_hist.append((x_update[0,0], x_update[1,0]))
        f_hist.append(f(x_update[0,0], x_update[1,0], *args))
        conv = convergence(f, x, x_update, *args) # determine convergence criteria
        x = x_update

    x_min = x_hist[-1]
    f_min = f(x_min[0], x_min[1], *args)

    return x_hist, f_hist, x_min, f_min


def sd(f, f_min, x, *args):
    
    epsilon = 10**(-4)
    
    def g(x, *args): 
        return f(x, *args) - (f_min + 0.5)

    solution = newton_raphson(g, x, epsilon, *args)
    return solution

def test(x, y, a, b):
	
	#return x**2 + y**2
	#return np.sin(x) + np.cos(y)
	return a * (np.cosh(b * x) + np.cosh(b * y))


if __name__ == "__main__":

	sol1 = grad_min(test, (-3., 1.5), 0.4, 2.0, .5)
	sol2 = newton_min(test, (-3., 1.5), 2.0, .5)
	x_hist1 = sol1[0]
	print(x_hist1)
	x_hist2 = sol2[0]
	print(x_hist2)
	x = np.linspace(-3., 3., 100)
	y = np.linspace(-3., 3., 100)
	Z = np.zeros((100, 100))
	
	for i, a in enumerate(x):
		for j, b in enumerate(y):
			Z[j, i] = test(a,b, 2.0, .5)
	
	X, Y = np.meshgrid(x,y)
	
	fig1, ax1 = plt.subplots()
	contour = ax1.contour(X, Y, Z)
	#contour = ax1.contourf(X, Y, Z, cmap=cm.viridis)
	fig1.colorbar(contour)
	ax1.plot([a[0] for a in x_hist1], [a[1] for a in x_hist1], color='green', 
             linestyle='--', marker='.', label='Gradient Method')
	ax1.plot([a[0] for a in x_hist2], [a[1] for a in x_hist2], color='red', 
             linestyle='--', marker='.', label='Newton\'s Method')
	ax1.grid()
	ax1.legend(loc='best')
	fig2, ax2 = plt.subplots()
	ax2.plot(range(len(x_hist1)), [a[0] for a in x_hist1], color='green', 
            linestyle='-', marker='.', label='Gradient Method')
	ax2.plot(range(len(x_hist2)), [a[0] for a in x_hist2], color='red', 
            linestyle='-', marker='.', label='Newton\'s Method')
	ax2.grid()
	ax2.legend(loc='best')
	plt.show()
	
	print('Gradient Method number of steps: {}'.format(len(x_hist1)))
	print('Newton\'s Method number of steps: {}'.format(len(x_hist2)))