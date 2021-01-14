from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
from sys import platform


bc = {0:'free', -1:'free'}

dim = 3
Np = 32
fil = activeFilament(dim = dim, Np = Np, radius = 1, b0 = 4, k = 10, S0 = 0, D0 = 0, bc = bc)

# fil.plotFilament(r = fil.r0)

fil.getSeparationVector()


print('separation vector',fil.dr_hat)


# Testing the Bond Angles Cython function
fil.filament.get_bond_angles(fil.dr_hat, fil.cosAngle)



cython_angles = fil.cosAngle

print('Bond angles (Cython)')
print(cython_angles)

fil.getBondAngles()

print('Bond angles (Python)')
print(fil.cosAngle)

plt.figure()
plt.plot(fil.cosAngle, 'gs', label ='pure python')
plt.plot(cython_angles, 'ro', alpha = 0.5, label ='cython')
plt.title('Bond angles')
plt.legend()
plt.show()
print('Mismatch in bending angles computation: {}'.format(np.sum((fil.cosAngle - cython_angles)**2)))


# Testing the connection forces function

print('Array send to cython fn', fil.F_conn)
# Compute using Cython function
fil.filament.connection_forces(fil.dr, fil.dr_hat, fil.F_conn) 

F_conn_cython = fil.F_conn

# Compute using Python function
fil.ConnectionForces()

# Compute using different (simpler) algorithm
F_conn_modified = fil.ConnectionForces_mod()

# Check that internal connection forces should sum to zero
for ii in range(dim):
	print('{} component of force summation (python): {}'.format(ii, np.sum(fil.F_conn[ii*Np:(ii+1)*Np])))
	print('{} component of force summation (python-modified algo): {}'.format(ii, np.sum(F_conn_modified[ii*Np:(ii+1)*Np])))
	print('{} component of force summation (cython): {}'.format(ii, np.sum(F_conn_cython[ii*Np:(ii+1)*Np])))
	

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
colors = ['r', 'g', 'b']
for ii in range(dim):
	ax[ii].plot(fil.F_conn[ii*Np:(ii+1)*Np], marker = 's', color = 'g')
	ax[ii].plot(F_conn_modified[ii*Np:(ii+1)*Np], marker = 'd', alpha = 0.4, color = 'b')
	ax[ii].plot(F_conn_cython[ii*Np:(ii+1)*Np], marker = 'o', alpha = 0.5, color = 'r')
	ax[ii].set_ylabel('F_{}'.format(ii))
	ax[ii].set_xlabel('Particle number')
plt.title('Connection Forces')
plt.show()

print('Mismatch in connection forces computation (Python vs Cython mod): {}'.format(np.sum((fil.F_conn - F_conn_cython)**2)))

print('Mismatch in connection forces computation (Python mod vs Cython): {}'.format(np.sum((F_conn_modified - F_conn_cython)**2)))

# # Testing the Tangent vectors function

# fil.filament.get_tangent_vectors(fil.dr_hat, fil.t_hat)

# tangent_vectors_cython = fil.t_hat

# fil.getTangentVectors()

# plt.figure()
# plt.plot(fil.t_hat, 'gs')
# plt.plot(tangent_vectors_cython, 'ro', alpha = 0.5)
# plt.title('Tangent vectors')
# plt.show()

# print('Error in tangent vector computation: {}'.format(np.sum(fil.t_hat - tangent_vectors_cython)))


# # Testing bending forces function
# fil.filament.bending_forces(fil.dr, fil.dr_hat, fil.cosAngle, fil.F_bending)

# F_bending_cython = fil.F_bending

# fil.BendingForces()

# plt.figure()
# plt.plot(fil.t_hat, 'gs')
# plt.plot(tangent_vectors_cython, 'ro', alpha = 0.5)
# plt.title('Bending forces')
# plt.show()
# print('Error in bending forces computation: {}'.format(np.sum(fil.F_bending - F_bending_cython)))




