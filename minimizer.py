import matplotlib.pyplot as plt
import numpy as np

def para_min(f, x): 
	'''
	para_min: Computes the minimum of the parabola formed by three points.
	Args: 
		f: function giving y coordinate of points.
		x: list of three points. 

	Returns: 
		x_min: x coord of minimum.
	'''

	num = 0.5*((x[2]**2-x[1]**2)*f(x[0])+(x[0]**2-x[2]**2)*f(x[1])+(x[1]**2-x[0]**2)*f(x[2]))
	denom = ((x[2]-x[1])*f(x[0])+(x[0]-x[2])*f(x[1])+(x[1]-x[0])*f(x[2]))
	x_min = num/denom
	return x_min


def minimise(f, start_points, epsilon): 
	'''
	minimise: Minimises a function using parabolic minimisation algorithm.
	Args: 
		f: function to be minimised. 
		start_points: list of three initial points. 
	'''
	
	points = zip(start_points, f(start_points))
	print points
	print [elem[0] for elem in points]
	#points_update = para_min(x)

	# x = [a for a in start_points]
	# y = [f(a) for a in start_points]

	# for i in range(2):
	# 	x_update = para_min(x)
	# 	x_min_list.append(x_update)
	# 	y_min_list.append(f(x_update))
	# 	x.append(x_update)
	# 	y.append(f(x_update))
	# 	x.sort()
	# 	x.pop()

	# conv = abs((x_min_list[-1]-x_min_list[-2])/x_min_list[-2])

	# while (conv>epsilon):
	# 	x_update = para_min(x)
	# 	x_min_list.append(x_update)
	# 	x.append(x_update)
	# 	x.sort()
	# 	x.pop()

	# return x

minimise(np.cosh, (-.5, .5, .1), 0.1)
x = np.linspace(-2,2)
plt.plot(x, np.cosh(x))
plt.grid()
plt.show()