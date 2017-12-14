# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from b1 import nll_2d
from min_2d import grad_min, newton_min, sd

f = open('lifetime.txt')  # measured lifetimes and uncertainty data
lifetime = []
uncertainty = []

for line in f:  # reading data line by line
    data = line.split()
    lifetime.append(float(data[0]))
    uncertainty.append(float(data[1]))

N = len(lifetime)  # number of data points

#start_point = (.4, .6) #(0.4, 0.6) works
start_point = (.46, 0.95)
sol1 = grad_min(nll_2d, start_point, 10**(-5), lifetime, uncertainty)
sol2 = newton_min(nll_2d, start_point, lifetime, uncertainty)
x_hist1 = sol1[0]
f_hist1 = sol1[1]

#print(sol1[0])
x_hist2 = sol2[0]
f_hist2 = sol2[1]
#print(sol2[0])
#
grid_points = 10
tau = np.linspace(0.38, 0.46, grid_points) # range for nll function
a = np.linspace(0.94, 1., grid_points)

nll_matrix = np.zeros((grid_points, grid_points)) # initialise array for nll function plotting
#gradient_matrix_x = np.zeros((grid_points, grid_points))
#gradient_matrix_y = np.zeros((grid_points, grid_points))
x_min = sol1[2]
f_min = sol2[3]
#print(sd(nll_2d, f_min, x_min[0]-0.01, x_min[1], lifetime, uncertainty))

a_range = np.arange(x_min[1] - 0.02, x_min[1] + 0.02, 10**(-3))
tau_contour1 = []
tau_contour2 = []

for i in a_range:
    tau_contour1.append(sd(nll_2d, f_min, x_min[0]-0.01, i, lifetime, uncertainty))
    tau_contour2.append(sd(nll_2d, f_min, x_min[0]+0.01, i, lifetime, uncertainty))

#fig, ax = plt.subplots()
#ax.plot(tau, [nll_2d(t, x_min[1]+0.01, lifetime, uncertainty)-(f_min+0.5) for t in tau])
#solution = newton_raphson(nll_2d, x_min[0]+0.01, 10**(-3), 

for i, t in enumerate(tau):
    print('\n')
    print('\r Filling 2D NLL function array (tau loop): {}/{}'.format(i, len(tau)), end="")
    print('\n')
    for j, b in enumerate(a):
        print('\r Filling 2D NLL function array (a loop): {}/{}'.format(j, len(a)), end="")
        #tqdm(tau, total=len(tau), desc='NLL array filling...'):
        nll_matrix[j, i] = nll_2d(tau=t, a=b, lifetime=lifetime, uncertainty=uncertainty)
        #x = np.asarray((t,b)).reshape((2,1))
        #gradient_matrix_x[j, i] = grad_cds(nll_2d, x, lifetime, uncertainty)[0,0]
        #gradient_matrix_y[j, i] = grad_cds(nll_2d, x, lifetime, uncertainty)[1,0]

#tau_contour, a_contour = sd(nll_2d, sol1[2], sol1[3], lifetime, uncertainty)


#gradient_np = np.gradient(nll_matrix)
#T, A = np.meshgrid(tau,a)
#
#fig1, ax1 = plt.subplots()
#contourf1 = ax1.contourf(T, A, gradient_np[1], cmap=cm.coolwarm)
#fig1.colorbar(contourf1)
#
#fig2, ax2 = plt.subplots()
#contourf2 = ax2.contourf(T, A, gradient_np[0], cmap=cm.coolwarm)
#fig2.colorbar(contourf2)
#
#fig3, ax3 = plt.subplots()
#contourf3 = ax3.contourf(T, A, gradient_matrix_x, cmap=cm.coolwarm)
#fig3.colorbar(contourf3)
#
#fig3, ax3 = plt.subplots()
#contourf3 = ax3.contourf(T, A, gradient_matrix_y, cmap=cm.coolwarm)
#fig3.colorbar(contourf3)
#plt.show()
##Plotting
fig1, ax1 = plt.subplots()
T, A = np.meshgrid(tau,a)

nll_2d_min, nll_2d_max = nll_matrix.min(), nll_matrix.max()

contours = ax1.contour(T, A, nll_matrix, 10, colours='black')
plt.clabel(contours, inline=True)

v = np.linspace(nll_2d_min, nll_2d_max, 10, endpoint=True)
contourf = ax1.contourf(T, A, nll_matrix, levels=v, cmap=cm.viridis, alpha=0.5)
#contour = ax1.contour(T, A, nll_matrix)
ax1.plot([a[0] for a in x_hist1], [a[1] for a in x_hist1], color='green', 
         linestyle='--', marker='.', label='Gradient Method')
#ax1.imshow(nll_matrix, extent=[tau.min(),tau.max(), a.min(), a.max()],
#           cmap='RdGy', alpha=.5)

ax1.plot([a[0] for a in x_hist2], [a[1] for a in x_hist2], color='red', 
         linestyle='--', marker='.', label='Newton\'s Method')

fig1.colorbar(contourf)
ax1.grid()
ax1.set_ylabel("a") 
ax1.set_xlabel("tau")
ax1.legend(loc='best')

fig2, ax2 = plt.subplots()
ax2.plot(range(len(x_hist1)), [a for a in f_hist1], color='green', linestyle='-', marker='.', label='Gradient Method')
ax2.plot(range(len(x_hist2)), [a for a in f_hist2], color='red', linestyle='-', marker='.', label='Newton\'s Method')
ax2.grid()
ax2.legend(loc='best')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(T, A, nll_matrix)
ax3.plot([a[0] for a in x_hist1], [a[1] for a in x_hist1], [a for a in f_hist1])    
ax3.plot([a[0] for a in x_hist2], [a[1] for a in x_hist2], [a for a in f_hist2])

fig4, ax4 = plt.subplots()
ax4.plot(tau_contour1, a_range, color='blue', linestyle=' ', marker='.')
ax4.plot(tau_contour2, a_range, color='blue', linestyle=' ', marker='.')
ax4.plot(x_min[0], x_min[1], color='red', linestyle=' ', marker='.')
ax4.grid()
ax4.set_ylabel("a") 
ax4.set_xlabel("tau")


plt.show()

print('Gradient Method number of steps: {}'.format(len(x_hist1)))
print('Gradient Method solution: {}'.format(x_hist1[-1]))
print('Newton\'s Method number of steps: {}'.format(len(x_hist2)))
print('Newton\'s Method solution: {}'.format(x_hist2[-1]))