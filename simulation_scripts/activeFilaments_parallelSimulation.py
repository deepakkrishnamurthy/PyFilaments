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
from pyfilaments._def import *


#-------------------------------------------------------------
# Filament parameters
#-------------------------------------------------------------
BC = {0:'clamped', -1:'free'}
NP = 32
K = 25
S0 = 0
# Activity profile parameters
activity_timescale = 750 # Activity time-scale (one compression and extension cycle)
duty_cycle = 0.5	# Relative time for compression relative to total activity time-scale
n_activity_cycles = 1 # No:of activity cycles we want to simulate
Tf = activity_timescale*n_activity_cycles # Total simulation time
time_step_save = 10
Npts = int(Tf/time_step_save) # No:of time points saved

def run_parametric_simulation(pid, parameter):
	# Filament stiffness sweep
	# fil = activeFilament(dim = 3, Np = 32, radius = radius, b0 = 2.1*radius, k = parameter, S0 = 0, D0 = 1.5, bc = bc)

	# Filamement length sweep
	# fil = activeFilament(dim = 3, Np = parameter, radius = radius, b0 = 2.1*radius, k = 25, S0 = 0, D0 = 1.5, bc = bc)

	# Filament activity sweep
	fil = activeFilament(dim = DIMS, Np = NP, radius = RADIUS, b0 = B0, k = K, S0 = S0, 
		D0 = parameter, bc = BC)

	fil.simulate(Tf, Npts, n_cycles = n_activity_cycles, save = True, overwrite = False, 
		path = ROOT_PATH, sim_type = 'point', init_condition = {'shape':'line'}, 
	activity={'type':'square-wave','activity_timescale':activity_timescale, 'duty_cycle':duty_cycle, 
	'start phase':0})


# parameter_list = np.linspace(0.5,3, 40)
parameter_list = 0

num_initial_conditions = 1

parameter_list_full = []

for ii in range(num_initial_conditions):
	parameter_list_full.append(parameter_list)

parameter_list_full = np.array(parameter_list_full).flatten()

print(parameter_list_full)

num_cores = multiprocessing.cpu_count()

num_cores = 12

results = Parallel(n_jobs=num_cores,  verbose=10)(delayed(run_parametric_simulation)(pid, parameter) for pid, parameter in enumerate(parameter_list_full))
