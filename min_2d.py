# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from helper import dot, lu

def convergence(f, x_old, x_new, *args):
    '''
    convergence: Determines convergence criteria for two consecutive points.
    Args:
        f: function.
        x_old: previous point. 
        x_new: next point. 
        *args: function arguments.
    Returns: 
        conv: convergence criteria.
    '''
    x_old = (x_old[0,0], x_old[1,0])
    x_new = (x_new[0,0], x_new[1,0])
    conv = (f(x_new[0], x_new[1], *args) - 
            f(x_old[0], x_old[1], *args)) / f(x_old[0], x_old[1], *args)
    
    return abs(conv)


def grad_fds(f, x, *args):
    '''
    grad_fds: Finds the gradient of a 2d function using the forward difference 
          scheme (FDS).
    Args: 
        f: function to take gradient of. 
        x: point to evaluate at.
        *args: function arguments. 
    Returns: 
        (dfdx, dfdy): gradient in tuple form.
        
    '''
    h = 10**(-5)
    dfdx = (f(x[0] + h, x[1]) - f(x[0], x[1])) / h
    dfdy = (f(x[0], x[1] + h) - f(x[0], x[1])) / h
    grad = np.zeros((2,1))
    grad[0,0] = dfdx
    grad[1,0] = dfdy
    
    return grad

def grad_bds(f, x, *args):
    '''
    grad_bds: Finds the gradient of a 2d function using the backwards difference 
          scheme (BDS).
    Args: 
        f: function to take gradient of. 
        x: point to evaluate at.
        *args: function arguments. 
    Returns: 
        (dfdx, dfdy): gradient in tuple form.
        
    '''
    h = 10**(-5)
    dfdx = (f(x[0], x[1]) - f(x[0] - h, x[1])) / h
    dfdy = (f(x[0], x[1]) - f(x[0], x[1] - h)) / h
    
    #return (dfdx, dfdy)
    grad = np.zeros((2,1))
    grad[0,0] = dfdx
    grad[1,0] = dfdy
    
    return grad

def grad_cds(f, x, *args):
    '''
    grad_cds: Finds the gradient of a 2d function using the central difference 
          scheme (CDS).
    Args: 
        f: function to take gradient of. 
        x: point to evaluate at as an (2,1) np.array.
        *args: function arguments. 
    Returns: 
        (dfdx, dfdy): gradient in tuple form.
        
    '''
    grad = np.zeros((2,1)) # empty array for gradient
    h = 10**(-5)
    x = (x[0,0], x[1,0]) # converts input np.array to a tuple to simplify code
    dfdx = (f(x[0] + h, x[1]) - f(x[0] - h, x[1]))/(2 * h)
    dfdy = (f(x[0], x[1] + h) - f(x[0], x[1] - h))/(2 * h)
    
    grad[0,0] = dfdx
    grad[1,0] = dfdy
    
    return grad

def hessian(f, x, *args):
    '''
    hessian: Finds the hessian of a 2d function using a finite difference scheme. 
    Args: 
        f: function to take gradient of. 
        x: point to evaluate at. 
        *args: function arguments.  
    Returns: 
        hess: hessian in np.array form. 
	'''
    
    hess = np.zeros((2,2))
    h = 10**(-5)
    x = (x[0,0], x[1,0]) # converts input np.array to a tuple to simplify code
    dfdxx = (f(x[0] + h, x[1]) - 2 * f(x[0], x[1]) + f(x[0] - h, x[1])) / (h**2)
    dfdyy = (f(x[0], x[1] + h) - 2 * f(x[0], x[1]) + f(x[0], x[1] - h)) / (h**2)
    dfdxy = (f(x[0] + h, x[1] + h) - f(x[0] + h, x[1] - h) - 
             f(x[0] - h, x[1] + h) + f(x[0] - h, x[1] - h)) / (4 * h**2) 
	
    hess[0,0] = dfdxx
    hess[0,1] = dfdxy
    hess[1,0] = dfdxy
    hess[1,1] = dfdyy
   
    return hess

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
    x = np.asarray(x).reshape((2,1)) # converts tuple to array so works with helper functions
    grad = grad_cds(f, x, *args) # calculate gradient
    x_update = x-a*grad # calcuate x_(n+1)
    x_hist.append((x[0,0], x[1,0])) # converts back to tuple for more readable form
    x_hist.append((x_update[0,0], x[1,0]))
    conv = convergence(f, x, x_update, *args) # determine convergence criteria
    x = x_update
    
    while (conv > 10**(-5)):
        grad = grad_cds(f, x, *args)
        x_update = x-a*grad
        x_hist.append((x_update[0,0], x[1,0])) #convert back to tuple for more readable form
        conv = convergence(f, x, x_update, *args) # determine convergence criteria
        x = x_update

    x_min = x_hist[-1]
    f_min = f(x_min[0], x_min[1], *args)
    
    return x_hist, x_min, f_min

def newton_min(f, x, *args):
    '''
    newton_min: Minimises a 2d function using Newton's method. 
    Args: 
        f: function to be minimized. 
        x: starting point. 
        a: step size. 
        *args: function arguments. 
    Returns: 
        x_hist: record of points in tuple form (x,y). 
        x_min: minimum coordinate.
        f_min: minimum value of function.
    '''
    
    x_hist = []
    x = np.asarray(x).reshape((2,1))
    grad = grad_cds(f, x, *args)
    hess = hessian(f, x, *args)
    inv_hess = lu(hess, np.identity(2))[2]
    x_hist.append((x[0,0], x[1,0]))
    d = dot(inv_hess,grad)
    x_update = x - d
    x_hist.append((x_update[0,0], x_update[1,0]))
    conv = convergence(f, x, x_update, *args) # determine convergence criteria
    x = x_update
    
    while (conv > 10**(-5)):
        grad = grad_cds(f, x, *args)
        hess = hessian(f, x, *args)
        inv_hess = lu(hess, np.identity(2))[2]
        d = dot(inv_hess,grad)
        x_update = x - d
        x_hist.append((x_update[0,0], x_update[1,0]))
        conv = convergence(f, x, x_update, *args) # determine convergence criteria
        x = x_update

    x_min = x_hist[-1]
    f_min = f(x_min[0], x_min[1], *args)
    
    return x_hist, x_min, f_min
    

def test(x, y):
    
    #return x**2 + y**2
    #return np.sin(x) + np.sin(y)
    return np.cosh(x) + np.cosh(y)


if __name__ == "__main__":

    sol1 = grad_min(test, (-12., 1.5), 0.1)
    sol2 = newton_min(test, (-12., 1.5))
    x_hist1 = sol1[0]
    print(x_hist1)
    x_hist2 = sol2[0]
    print(x_hist2)
    x = np.linspace(-3., 3., 100)
    y = np.linspace(-3., 3., 100)
    Z = np.zeros((100, 100))
    
    for i, a in enumerate(x): 
        for j, b in enumerate(y): 
            Z[j, i] = test(a,b)
    
    X, Y = np.meshgrid(x,y)
    
    fig1, ax1 = plt.subplots()
    contour = ax1.contour(X, Y, Z)
    #contour = ax1.contourf(X, Y, Z, cmap=cm.viridis)
    fig1.colorbar(contour)
    ax1.plot([a[0] for a in x_hist1], [a[1] for a in x_hist1], color='green', linestyle='--', marker='.', label='Gradient Method')
    ax1.plot([a[0] for a in x_hist2], [a[1] for a in x_hist2], color='red', linestyle='--', marker='.', label='Newton\'s Method')
    ax1.grid()
    ax1.legend(loc='best')
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(x_hist1)), [a[0] for a in x_hist1], color='green', linestyle='-', marker='.', label='Gradient Method')
    ax2.plot(range(len(x_hist2)), [a[0] for a in x_hist2], color='red', linestyle='-', marker='.', label='Newton\'s Method')
    ax2.grid()
    ax2.legend(loc='best')
    plt.show()
    
    print('Gradient Method number of steps: {}'.format(len(x_hist1)))
    print('Newton\'s Method number of steps: {}'.format(len(x_hist2)))

   # print(hessian(test, (1,1)))