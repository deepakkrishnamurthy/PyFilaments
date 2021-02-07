from pyfilaments.activeFilaments import activeFilament
import numpy as np
import matplotlib.pyplot as plt 


bc = {0:'free', -1:'free'}

dim = 3
Np = 32

fil = activeFilament(dim = dim, Np = Np, radius = 1, b0 = 4, k = 10, S0 = 0, D0 = 0, bc = bc)
fil.plotFilament(r = fil.r)


fil.get_separation_vectors()
fil.get_tangent_vectors()
fil.t_hat_array = fil.reshape_to_array(fil.t_hat)
fil.p = fil.t_hat_array

print(50*'*')
print('Separation vectors',fil.dr_hat)
print(50*'*')

# Testing the Bond Angles Cython function
fil.filament.get_bond_angles(fil.dr_hat, fil.cosAngle)

cython_angles = fil.cosAngle


plt.figure()
# plt.plot(fil.cosAngle, 'gs', label ='pure python')
plt.plot(cython_angles, 'ro', alpha = 0.5, label ='cython')
plt.title('Bond angles')
plt.legend()
plt.show()


fil.filament.connection_forces(fil.dr, fil.dr_hat, fil.F_conn) 

F_conn_cython = fil.F_conn

# Check that internal connection forces should sum to zero
for ii in range(dim):
	print('{} component of force summation (cython): {}'.format(ii, np.sum(F_conn_cython[ii*Np:(ii+1)*Np])))
	

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
colors = ['r', 'g', 'b']
for ii in range(dim):
	# ax[ii].plot(fil.F_conn[ii*Np:(ii+1)*Np], marker = 's', color = 'g')
	ax[ii].plot(F_conn_cython[ii*Np:(ii+1)*Np], marker = 'o', alpha = 0.5, color = 'r')
	ax[ii].set_ylabel('F_{}'.format(ii))
	ax[ii].set_xlabel('Particle number')
plt.title('Connection Forces')
plt.show()

# Testing bending forces function
fil.filament.bending_forces(fil.dr, fil.dr_hat, fil.cosAngle, fil.F_bending)

F_bending_cython = fil.F_bending


fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
colors = ['r', 'g', 'b']
for ii in range(dim):
	# ax[ii].plot(F_bending_python[ii,:], marker = 's', color = 'g', label = 'python')
	ax[ii].plot(F_bending_cython[ii,:], marker = 'o', alpha = 0.4, color = 'r', label = 'cython')

	ax[ii].set_ylabel('bending forces_{}'.format(ii))
	ax[ii].set_xlabel('Particle number')

plt.show()

for ii in range(dim):
	print('{} component of bending force summation (cython): {}'.format(ii, np.sum(F_bending_cython[ii,:])))
