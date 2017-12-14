# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import rc
from pylab import rcParams

def figsize(scale):
    fig_width_pt = 469.755                         
    inches_per_pt = 1.0/72.27 
    figwidth = fig_width_pt*inches_per_pt*scale
    figheight = figwidth
    return figwidth, figheight

def latexfigure(scale):
    rc('font', **{'family':'sans-serif','sans-serif':['Computer Modern Sans serif'], 'size':9}) #
    rc('text', usetex=True)
    rcParams['figure.figsize'] = figsize(scale)[0], figsize(scale)[1]
    
def dot(a,b):

	'''Multiplies two matrices if their dimensions are appropriate '''

	a_nrows = a.shape[0]
	a_ncols = a.shape[1]
	b_nrows = b.shape[0]
	b_ncols = b.shape[1]

	if(a_ncols!=b_nrows):
		raise Exception('Matrices cannot be multiplied')

	c = np.zeros((a_nrows,b_ncols))

	for i in range(0,a_nrows):
		for j in range(0,b_ncols):
			c[i,j] = sum([x*y for x,y in zip(a[i,:],b[:,j])]) #Gets required row from a and column from b and sums over product of each element.

	return c

def crout(a):

	'''
	crout: Carries out LU decomposition using Crout's algorithm.
	Args: 
		a: Input matrix to be decomposed. Assumed to be a square np.array. 
	Returns (as a tuple): 	
		l: Lower diagonal portion of decomposed matrix.
		u: Upper diagonal portion of decomposed matrix.
		det(a): Determinant of the original matrix a.
	'''

	a_nrows = a.shape[0] # checks # of rows and columns
	a_ncols = a.shape[1]

	if(a_ncols!=a_nrows): # throws exception if matrix is not square
		raise Exception('LU Decomposition requires a square matrix')

	N = a_nrows

	l = np.zeros((N,N)) # initialises lower and upper diagonal matrices
	u = np.zeros((N,N))

	for i in range(N): # sets the diagonal of the lower matrix to 1 according to Crout's alg. convention.
		l[i,i]=1.

	for j in range(N): # loops through the matrix columns
		for i in range(j+1): # loops through the rows of the upper diagonal matrix, u 
			if(i==0): 
				sumterm = 0. # sets the summation term to zero for the first row
			else: 
				sumterm = 0.
				for k in range(i): # calculates the summation term for 2nd row onwards
					sumterm+=l[i,k]*u[k,j]

			u[i,j] = a[i,j]-sumterm # evalutes for the required element of the upper diagonal matrix
		
		for i in range(j+1, N): # loops through the rows of the lower diagonal matrix, l
			if(j==0): 
				sumterm = 0. # sets the summation term to zero for the first row
			else: 
				sumterm = 0.
				for k in range(j): # calculates the summation term for the second row onwards
					sumterm+=l[i,k]*u[k,j]

			l[i,j]=(1./u[j,j])*(a[i,j]-sumterm) # evaluates for the required element of the lower diagonal matrix

	det = 1 

	for i in range(N): 
		det *= u[i,i] # calculates the determinant of a

	if(np.allclose(a,np.dot(l,u))!=True): # checks the decomposition using numpy's matrix multiplication method
		print('Error in decomposition.')

	return l,u,det

def lu(a,b): 

	'''
	lu: Carries out forward & backward substitution in order to solve the matrix equation a.x=b, after 
	decomposition through Crout's algorithm. 

	Args:
		a: Coefficient matrix for equation system. Assumed to be a square NxN numpy.array.
		b: Matrix or vector of constants for equation system. Assumed to be a 2D numpy.array with size (N,1) if representing a column vector,
		and with size (N,M) if representing M equation systems.
	Returns: 
		x: Solution matrix or vector. 
		det: Determinant of matrix a.
	'''

	N = a.shape[0] # no of rows in a
	M = b.shape[1] # no of columns in b

	result = crout(a) # carries out lu decomposition of a via crout's algorithm
	l = result[0]
	u = result[1]
	det = result[2]

	y = np.zeros((N,M)) # initialises y & x matrices (column vectors for single equation system)
	x = np.zeros((N,M))

	
	for k in range(M): # loops over each equation system (no of columns in x)

		y[0,k] = b[0,k]/l[0,0] # first row of y (for a given column if multiple equation systems)

		for i in range(1,N): # forward substitution to determine y
			sumterm = 0
			for j in range(0,i): # loops over each column in the lower diagonal matrix
				sumterm+=l[i,j]*y[j,k] # evaluates the summation term
			y[i,k] = (1./l[i,i])*(b[i,k]-sumterm) # determines the required row of y (for a given column if multiple equation systems)

		x[N-1] = y[N-1]/u[N-1,N-1] # last row of x (for a given column if multiple equation systems)

		for i in range(N-2,-1,-1): # backward substitution to determine x
			sumterm = 0
			for j in range(i,N): # loops over each column in the upper diagonal matrix
				sumterm+=u[i,j]*x[j,k] # evaluates the summation term for this row
			x[i,k] = (1./u[i,i])*(y[i,k]-sumterm) # determines the required row of x (for a given column if multiple equation systems)

	if(np.allclose(b,np.dot(a,x))!=True): # checks the solution using numpy's matrix multiplication method 
		print('Solution error.')

	return l,u,x,det

def grad(f, x, *args):
	'''
	grad: Finds the gradient of a 1d function using the central difference 
		  scheme (CDS).
	Args: 
		f: function to take gradient of. 
		x: point to evaluate at.
		*args: function arguments. 
	Returns: 
		dfdx: gradient.
		
	'''
	h = 10**(-7)
	dfdx = (f(x + h, *args) - f(x - h, *args)) / (2 * h)
	
	return dfdx


def newton_raphson(f, x, epsilon, *args): 
    '''
    newton_raphson: 1d root finder
    Args:
        f: function to find root.
        x: initial guess.
        epsilon: desired accuracy of solution.
        *args: function arguments. 
    Returns: 
        root: root of function.
    '''
    x_update = x - (f(x, *args) / grad(f, x, *args))
    conv = abs((x_update-x)/x_update)
    x = x_update
    i = 0
    while (conv > epsilon):
        if (i<20): 
            x_update = x - (f(x, *args) / grad(f, x, *args))
            conv = abs((x_update-x)/x_update)
            x = x_update
            i+=1
        else:
            x = float('nan')
            break
    root = x

    return root

def convergence(f, x_old, x_new, *args):
	'''
	convergence: Determines convergence criteria for two consecutive points.
	Args:
		f: function.
		x_old: previous point in np array form. 
		x_new: next point in np array form. 
		*args: function arguments.
	Returns: 
		conv: convergence criteria.
	'''
	x_old = (x_old[0,0], x_old[1,0])
	x_new = (x_new[0,0], x_new[1,0])
	conv = (f(x_new[0], x_new[1], *args) - 
			 f(x_old[0], x_old[1], *args)) / f(x_old[0], x_old[1], *args)
	
	return abs(conv)


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
	h = 10**(-7)
	x = (x[0,0], x[1,0]) # converts input np.array to a tuple to simplify code
	dfdx = (f(x[0] + h, x[1], *args) - f(x[0] - h, x[1], *args)) / (2 * h)
	dfdy = (f(x[0], x[1] + h, *args) - f(x[0], x[1] - h, *args)) / (2 * h)
	
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
	dfdxx = (f(x[0] + h, x[1], *args) - 2 * f(x[0], x[1], *args) + 
            f(x[0] - h, x[1], *args)) / (h**2)
	dfdyy = (f(x[0], x[1] + h, *args) - 2 * f(x[0], x[1], *args) + 
            f(x[0], x[1] - h, *args)) / (h**2)
	dfdxy = (f(x[0] + h, x[1] + h, *args) - f(x[0] + h, x[1] - h, *args) - 
			   f(x[0] - h, x[1] + h, *args) + f(x[0] - h, x[1] - h, *args)) / (4 * h**2) 
	
	hess[0,0] = dfdxx
	hess[0,1] = dfdxy
	hess[1,0] = dfdxy
	hess[1,1] = dfdyy
   
	return hess