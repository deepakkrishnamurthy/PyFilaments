# test code to compare speeds of Python vs Cython sub-routines

from pyfilaments.activeFilaments import activeFilament
import numpy as np
import matplotlib.pyplot as plt 
import time
bc = {0:'free', -1:'free'}

dim = 3
Np = 32

Nt = 1000

n_conditions = 8
python_times = np.zeros(n_conditions)
cython_times = np.zeros(n_conditions)
Np_array = np.zeros(n_conditions)
pystokes_times = np.zeros(n_conditions)
for index in range(n_conditions):

	Np = 32*(index+1)
	Np_array[index] = Np

	fil = activeFilament(dim = dim, Np = Np, radius = 1, b0 = 4, k = 10, S0 = 0, D0 = 0, bc = bc)
	# t1 = time.time()
	# for ii in range(Nt):
	# 	fil.getSeparationVector()

	# t_separation_vector_python = 1e6*(time.time()-t1)/Nt
	# print('Time taken per call (us): {} over {} calls (Python)'.format(t_separation_vector_python, Nt))

	fil.getSeparationVector()

	print(50*'-')
	print('Apply BC velocity')
	print(50*'-')
	t1 = time.time()
	for ii in range(Nt):
		fil.ApplyBC_velocity()

	t_apply_bc = 1e6*(time.time()-t1)/Nt
	print('Time taken per call (us): {} over {} calls (Python)'.format(t_apply_bc, Nt))


	# print(50*'-')
	# print('Tangent vectors function')
	# print(50*'-')
	# t1 = time.time()
	# for ii in range(Nt):
	# 	fil.getTangentVectors()

	# t_tangent_vector_python = 1e6*(time.time()-t1)/Nt
	# print('Time taken per call (us): {} over {} calls (Python)'.format(t_tangent_vector_python, Nt))

	# # Reshape to array function
	# print(50*'-')
	# print('Reshape function')
	# print(50*'-')
	# t1 = time.time()
	# for ii in range(Nt):
	# 	fil.reshapeToArray(fil.t_hat)

	# t_reshape_to_array_python = 1e6*(time.time()-t1)/Nt
	# print('Time taken per call (us): {} over {} calls (Python)'.format(t_reshape_to_array_python, Nt))

	# # Set Activity Forces
	# print(50*'-')
	# print('Set Activity Forces')
	# print(50*'-')
	# t1 = time.time()
	# for ii in range(Nt):
	# 	fil.setActivityForces(t = 0)


	# t_activity_forces_python = 1e6*(time.time()-t1)/Nt
	# print('Time taken per call (us): {} over {} calls (Python)'.format(t_activity_forces_python, Nt))


	# Set stresselt and potDipole

	# print(50*'-')
	# print('Set Stresslet')
	# print(50*'-')
	# t1 = time.time()
	# for ii in range(Nt):
	# 	fil.setStresslet()
	# 	fil.setPotDipole()


	# t_set_stresslet_python = 1e6*(time.time()-t1)/Nt
	# print('Time taken per call (us): {} over {} calls (Python)'.format(t_set_stresslet_python, Nt))


	# PyStokes functions
	# print(50*'-')
	# print('Pystokes and Pyforces functions')
	# print(50*'-')

	# t1 = time.time()
	# for ii in range(Nt):

	# 	fil.rm.stokesletV(fil.drdt, fil.r, fil.F)
	# 	fil.ff.lennardJones(fil.F, fil.r, fil.ljeps, fil.ljrmin)
	# 	fil.rm.stressletV(fil.drdt, fil.r, fil.S)	
	# 	fil.rm.potDipoleV(fil.drdt, fil.r, fil.D)

	# t_pystokes = 1e6*(time.time()-t1)/Nt
	# print('Time taken per call (us): {} over {} calls (Python)'.format(t_pystokes, Nt))

	# pystokes_times[index] = t_pystokes

	# Bond angles function
	# print(50*'-')
	# print('Bond angles function')
	# print(50*'-')
	# # Python
	# t1 = time.time()
	# for ii in range(Nt):
	# 	fil.getBondAngles()

	# t_bond_angles_python = 1e6*(time.time()-t1)/Nt
	# print('Time taken per call (us): {} over {} calls (Python)'.format(t_bond_angles_python, Nt))

	# # Cython
	# t1 = time.time()
	# for ii in range(Nt):
	# 	fil.filament.get_bond_angles(fil.dr_hat, fil.cosAngle)
	
	# t_bond_angles_cython = 1e6*(time.time()-t1)/Nt
	# print('Time taken per call (us): {} over {} calls (Cython)'.format(t_bond_angles_cython, Nt))

	# print(50*'-')
	# print('Connection Forces function')
	# print(50*'-')
	# # Python
	# t1 = time.time()
	# for ii in range(Nt):
	# 	fil.ConnectionForces()

	# t_connection_forces_python = 1e6*(time.time()-t1)/Nt
	# print('Time taken per call (us): {} over {} calls (Python)'.format(t_connection_forces_python, Nt))

	# # Cython
	# t1 = time.time()
	# for ii in range(Nt):
	# 	fil.filament.connection_forces(fil.dr, fil.dr_hat, fil.F_conn) 
	# t_connection_forces_cython = 1e6*(time.time()-t1)/Nt
	# print('Time taken per call (us): {} over {} calls (Cython)'.format(t_connection_forces_cython, Nt))

	# print(50*'-')
	# print('Bending Forces function')
	# print(50*'-')
	# # Python
	# t1 = time.time()
	# for ii in range(Nt):
	# 	fil.BendingForces()
	# t_bending_forces_python = 1e6*(time.time()-t1)/Nt
	# print('Time taken per call (us): {} over {} calls (Python)'.format(t_bending_forces_python, Nt))

	# # Cython
	# t1 = time.time()
	# for ii in range(Nt):
	# 	fil.filament.bending_forces(fil.dr, fil.dr_hat, fil.cosAngle, fil.F_bending)
	
	# t_bending_forces_cython = 1e6*(time.time()-t1)/Nt
	# print('Time taken per call (us): {} over {} calls (Cython)'.format(t_bending_forces_cython, Nt))

	# python_times[index] = t_bond_angles_python + t_connection_forces_python + t_bending_forces_python
	# cython_times[index] = t_bond_angles_cython + t_connection_forces_cython + t_bending_forces_cython




# Plot
# plt.figure()
# plt.plot(Np_array, python_times, marker = 'o', color = 'g', label = 'Python')
# plt.plot(Np_array, cython_times, marker = 's', color = 'r', label = 'Cython (with parallelization)')
# plt.xlabel('No:of particles (Np)')
# plt.ylabel('Code execution time (core functions) (us)')
# plt.xticks(ticks = Np_array)

# plt.title('Code execution time (core functions) over {} calls vs Np'.format(Nt))
# plt.legend()
# plt.show(block = False)

# # Plot
# plt.figure()
# # plt.plot(Np_array, python_times, marker = 'o', color = 'g', label = 'Python')
# plt.plot(Np_array, cython_times, marker = 's', color = 'r', label = 'Cython (with parallelization)')
# plt.xticks(ticks = Np_array)

# plt.xlabel('No:of particles (Np)')
# plt.ylabel('Code execution time (core functions) (us)')
# plt.title('Code execution time (core functions) over {} calls vs Np'.format(Nt))
# plt.legend()
# plt.show()

# plt.figure()
# # plt.plot(Np_array, python_times, marker = 'o', color = 'g', label = 'Python')
# plt.plot(Np_array, python_times/cython_times, marker = '^', color = 'b')
# plt.xticks(ticks = Np_array)

# plt.xlabel('No:of particles (Np)')
# plt.ylabel('Speed-up factor')
# plt.title('Code execution time speed-up for Cython vs Python (with parallelization)')
# plt.legend()
# plt.show()

# plt.figure()
# # plt.plot(Np_array, python_times, marker = 'o', color = 'g', label = 'Python')
# plt.plot(Np_array, pystokes_times, marker = '^', color = 'm')
# plt.xticks(ticks = Np_array)

# plt.xlabel('No:of particles (Np)')
# plt.ylabel('Pystokes functions excecution times (us)')
# plt.title('Pystokes functions excecution times')
# plt.legend()
# plt.show()

# import pandas as pd

# df = pd.DataFrame({'Np':Np_array, 'Pystokes times (us)':pystokes_times})

# df.to_csv('Pystokes_WithParallelization.csv')
