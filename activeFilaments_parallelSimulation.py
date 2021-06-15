''' Setup and run an active-filament simulation parametric sweep (using parallelization) and save the data.

'''
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
from sys import platform
from joblib import Parallel, delayed
import multiprocessing

from pyfilaments.activeFilaments import activeFilament

# Check which platform
if platform == "linux" or platform == "linux2":
	print("linux system")
	# root_path = '/home/deepak/LacryModelling_Local/SimulationData'
	root_path = '/home/deepak/Dropbox/LacryModeling/ModellingResults'
	

elif platform == 'darwin':
	print("OSX system")
	root_path = '/Users/deepak/Dropbox/LacryModeling/'

# Activity profile parameters
activity_timescale = 1500
duty_cycle = 0.5

# No:of activity cycles we want to simulate
n_activity_cycles = 500
# Total simulation time
Tf = activity_timescale*n_activity_cycles

# activity_timescale = 1000

# Total simulation time
# No:of time points saved
time_step_save = 10
Npts = int(Tf/time_step_save)

bc = {0:'clamped', -1:'free'}

def run_parametric_simulation(pid, parameter):
	radius = 1
	fil = activeFilament(dim = 3, Np = 32, radius = radius, b0 = 2.1*radius, k = parameter, S0 = 0, D0 = 1.5, bc = bc)

	fil.simulate(Tf, Npts, n_cycles = n_activity_cycles, save = True, overwrite = False, path = root_path, sim_type = 'point', 
	init_condition = {'shape':'line'}, 
	activity={'type':'normal','activity_timescale':activity_timescale, 'duty_cycle':duty_cycle, 'noise_scale':0.1})


parameter_list = np.array([15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 45])
num_initial_conditions = 1

parameter_list_full = []

for ii in range(num_initial_conditions):
	parameter_list_full.append(parameter_list)

parameter_list_full = np.array(parameter_list_full).flatten()

print(parameter_list_full)

num_cores = multiprocessing.cpu_count()

num_cores = 12

results = Parallel(n_jobs=num_cores,  verbose=10)(delayed(run_parametric_simulation)(pid, parameter) for pid, parameter in enumerate(parameter_list_full))
