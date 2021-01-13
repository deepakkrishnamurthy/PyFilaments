from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
from sys import platform
import pandas as pd
import time


# Load simulation parameters file

sim_parameters = pd.read_csv('simulation_parameters.csv')

for ii in range(len(sim_parameters)):

	activity_timescale = sim_parameters['activity_timescale'][ii]

	# activity_timescale = 1000
	activityFreq = 1.0/activity_timescale

	print('Activity frequency: {}'.format(activityFreq))

	# Total simulation time
	Tf = activity_timescale*sim_parameters['sim_time_scalefactor'][ii]

	print('Total simulation time: {}'.format(Tf))

	# No:of time points saved
	Npts = int(Tf/10)

	t_array = np.linspace(0, Tf+10, Npts)

	activity_profile = -signal.square(2*np.pi*activityFreq*t_array)

	activity_Function =  interpolate.interp1d(t_array, activity_profile)

	plt.style.use('dark_background')
	# plt.figure()
	# plt.plot(t_array, activity_profile)
	# plt.show(block=False)

	bc = {0:'clamped', -1:'free'}

	# N-simulation with the same conditions (different random initial conditions)
	N_simulations = sim_parameters['N_simulations'][ii]

	print(N_simulations)

	for jj in range(N_simulations):
		fil = activeFilament(dim = 3, Np = sim_parameters['Np'][ii], radius = sim_parameters['radius'][ii], b0 = sim_parameters['b0'][ii], kappa_hat = sim_parameters['kappa_hat'][ii], S0 = sim_parameters['S0'][ii], D0 = sim_parameters['D0'][ii], bc = bc, bending_axial_scalefactor = sim_parameters['bending_scale_factor'][ii])

		# fil.plotFilament(r = fil.r0)




		# Check which platform
		if platform == "linux" or platform == "linux2":
			print("linux system")
			root_path = '/home/deepak/Dropbox/LacryModeling/ModellingResults'
			

		elif platform == 'darwin':
			print("OSX system")
			root_path = '/Users/deepak/Dropbox/LacryModeling/ModellingResults'
			

		# elif platform == "win32":
		# 	print("Windows")
		# 	with open(os.devnull, 'w') as fnull:
		# 		exit_code = subprocess.call([compiler, '-Xpreprocessor', '-fopenmp', '-lomp', filename], stdout=fnull, stderr=fnull)


		fil.simulate(Tf, Npts, activity_profile = activity_Function, save = True, overwrite = False, path = root_path ,
					activity_timescale = activity_timescale, sim_type = 'point', init_condition = {'shape':'line'})


