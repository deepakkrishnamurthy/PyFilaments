from pyfilaments.activeFilaments import activeFilament
import numpy as np
import matplotlib.pyplot as plt 


bc = {0:'free', -1:'free'}

dim = 3
Np = 32


N_time = 1

error_bond_angles = []
error_tangent_vectors = []
error_connection_forces = []
error_bending_forces = []

for time_point in range(N_time):

	fil = activeFilament(dim = dim, Np = Np, radius = 1, b0 = 4, k = 10, S0 = 0, D0 = 0, bc = bc)
	fil.plotFilament(r = fil.r)


	fil.get_separation_vectors()
	fil.get_tangent_vectors()
	fil.t_hat_array = fil.reshape_to_array(fil.t_hat)
	fil.p = fil.t_hat_array

	# Testing the Bond Angles Cython function
	fil.filament.get_bond_angles(fil.dr_hat, fil.cosAngle)

	cython_angles = fil.cosAngle

	fil.get_bond_angles()

	# plt.figure()
	# plt.plot(fil.cosAngle, 'gs', label ='pure python')
	# plt.plot(cython_angles, 'ro', alpha = 0.5, label ='cython')
	# plt.title('Bond angles')
	# plt.legend()
	# plt.show(block = False)
	# print('Mismatch in bending angles computation: {}'.format(np.sum((fil.cosAngle - cython_angles)**2)))

	error_bond_angles.append(np.sum((fil.cosAngle - cython_angles)**2))

	# Testing the tangent vectors function

	# fil.filament.get_tangent_vectors(fil.dr_hat, fil.t_hat)
	# t_hat_cython = fil.t_hat

	# fil.getTangentVectors()
	# t_hat_python = fil.t_hat

	# fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
	# colors = ['r', 'g', 'b']
	# for ii in range(dim):
	# 	ax[ii].plot(t_hat_python[ii,:], marker = 's', color = 'g')
	# 	ax[ii].plot(t_hat_cython[ii,:], marker = 'o', alpha = 0.5, color = 'r')
	# 	ax[ii].set_ylabel('t_hat_{}'.format(ii))
	# 	ax[ii].set_xlabel('Particle number')
	# plt.title('Tangent vectors')
	# plt.show()

	# print('Mismatch in tangent vectors computation (Python vs Cython mod): {}'.format(np.sum((t_hat_python - t_hat_cython)**2)))



	# Testing the connection forces function
	# Compute using Cython function
	fil.filament.connection_forces(fil.dr, fil.dr_hat, fil.F_conn) 

	F_conn_cython = fil.F_conn

	# Compute using Python function
	fil.connection_forces()


	# Check that internal connection forces should sum to zero
	for ii in range(dim):
		print('{} component of force summation (python): {}'.format(ii, np.sum(fil.F_conn[ii*Np:(ii+1)*Np])))
		print('{} component of force summation (cython): {}'.format(ii, np.sum(F_conn_cython[ii*Np:(ii+1)*Np])))
		

	fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
	colors = ['r', 'g', 'b']
	for ii in range(dim):
		ax[ii].plot(fil.F_conn[ii*Np:(ii+1)*Np], marker = 's', color = 'g')
		ax[ii].plot(F_conn_cython[ii*Np:(ii+1)*Np], marker = 'o', alpha = 0.5, color = 'r')
		ax[ii].set_ylabel('F_{}'.format(ii))
		ax[ii].set_xlabel('Particle number')
	plt.title('Connection Forces')
	plt.show()

	print('Mismatch in connection forces computation (Python vs Cython mod): {}'.format(np.sum((fil.F_conn - F_conn_cython)**2)))


	error_connection_forces.append(np.sum((fil.F_conn - F_conn_cython)**2))

	# Testing bending forces function
	fil.filament.bending_forces(fil.dr, fil.dr_hat, fil.cosAngle, fil.F_bending)

	F_bending_cython = fil.F_bending

	fil.bending_forces()
	F_bending_python = fil.F_bending


	fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
	colors = ['r', 'g', 'b']
	for ii in range(dim):
		ax[ii].plot(F_bending_python[ii,:], marker = 's', color = 'g', label = 'python')
		ax[ii].plot(F_bending_cython[ii,:], marker = 'o', alpha = 0.4, color = 'r', label = 'cython')

		ax[ii].set_ylabel('bending forces_{}'.format(ii))
		ax[ii].set_xlabel('Particle number')

	plt.show()

	print('Mismatch in bending forces computation: {}'.format(np.sum((F_bending_cython - F_bending_python)**2)))

	# Check that internal connection forces should sum to zero
	for ii in range(dim):
		print('{} component of bending force summation (python): {}'.format(ii, np.sum(F_bending_python[ii,:])))
		print('{} component of bending force summation (cython): {}'.format(ii, np.sum(F_bending_cython[ii,:])))
		
	error_bending_forces.append(np.sum((F_bending_cython - F_bending_python)**2))



# plt.figure()
# plt.plot(error_bond_angles)
# plt.title('Bond angles error')
# plt.show(block = False)

# plt.figure()
# plt.plot(error_tangent_vectors)
# plt.title('Tangent vectors error')
# plt.show(block = False)

# plt.figure()
# plt.plot(error_connection_forces)
# plt.title('Connection forces error')
# plt.show(block = False)

# plt.figure()
# plt.plot(error_bending_forces)
# plt.title('Bending forces error')
# plt.show()


