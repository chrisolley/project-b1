# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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
    dfdx = (f(x[0] + h, x[1]) - f(x[0], x[1]))/h
    dfdy = (f(x[0], x[1] + h) - f(x[0], x[1]))/h
    
    return (dfdx, dfdy)

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
    dfdx = (f(x[0], x[1]) - f(x[0] - h, x[1]))/h
    dfdy = (f(x[0], x[1]) - f(x[0], x[1] - h))/h
    
    return (dfdx, dfdy)

def grad_cds(f, x, *args):
    '''
    grad_cds: Finds the gradient of a 2d function using the central difference 
          scheme (CDS).
    Args: 
        f: function to take gradient of. 
        x: point to evaluate at.
        *args: function arguments. 
    Returns: 
        (dfdx, dfdy): gradient in tuple form.
        
    '''
    h = 10**(-5)
    dfdx = (f(x[0] + h, x[1]) - f(x[0] - h, x[1]))/(2 * h)
    dfdy = (f(x[0], x[1] + h) - f(x[0], x[1] - h))/(2 * h)
    
    return (dfdx, dfdy)

def grad_min(f, x, a, *args):
    '''
    grad_min: Minimises a 2d function using the gradient method. 
    Args: 
        f: function to be minimized. 
        x: starting point. 
        a: step size. 
        *args: function arguments. 
    Returns: 
        x_hist: record of points. 
        x_min: minimum coordinate.
        f_min: minimum value of function.
        
    '''
    x_hist = []
    grad = grad_cds(f, x, *args)
    x_update = tuple(p-a*q for p,q in zip(x, grad))
    x_hist.append(x)
    x_hist.append(x_update)
    conv = (f(x_update[0], x_update[1], *args)
            - f(x[0], x[1], *args)) / f(x[0], x[1], *args)
    x = x_update
    
    while (abs(conv) > 10**(-5)):
        grad = grad_cds(f, x, *args)
        x_update = tuple(p - a * q for p, q in zip(x, grad))
        conv = (f(x_update[0], x_update[1], *args) - 
                f(x[0], x[1], *args)) / f(x[0], x[1], *args)
        x_hist.append(x_update)
        x = x_update

    x_min = x_hist[-1]
    f_min = f(x_min[0], x_min[1], *args)
    
    return x_hist, x_min, f_min

def test(x, y):
    
    #return x**2 + y**2
    return np.sin(x) + np.sin(y)

if __name__ == "__main__":

    sol = grad_min(test, (3., 1.6), .5)
    
    x_hist = sol[0]
    
    x = np.linspace(0, 2. * np.pi, 100)
    y = np.linspace(0, 2. * np.pi, 100)
    Z = np.zeros((100, 100))
    
    for i, a in enumerate(x): 
        for j, b in enumerate(y): 
            Z[j, i] = test(a,b)
    
    X, Y = np.meshgrid(x,y)
    
    fig, ax = plt.subplots()
    contour = ax.contour(X, Y, Z)#, cmap=cm.viridis) 
    fig.colorbar(contour)
    ax.plot([a[0] for a in x_hist], [a[1] for a in x_hist], color='red', linestyle='', marker='.')
    plt.show()