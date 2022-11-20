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

def run_parametric_simulation(pid, parameter):
	# Filament stiffness sweep
	# fil = activeFilament(dim = 3, Np = 32, radius = radius, b0 = 2.1*radius, k = parameter, S0 = 0, D0 = 1.5, bc = bc)

	# Filamement length sweep
	# fil = activeFilament(dim = 3, Np = parameter, radius = radius, b0 = 2.1*radius, k = 25, S0 = 0, D0 = 1.5, bc = bc)

	# Filament activity sweep
	# fil = activeFilament(dim = DIMS, Np = NP, radius = RADIUS, b0 = B0, k = K, S0 = S0, 
	# 	D0 = parameter, bc = BC)

		# fil.simulate(Tf, Npts, save = True, overwrite = False, 
	# 	path = ROOT_PATH, sim_type = 'dist', init_condition = {'shape':'line'}, 
	# 	activity = activity_parameters)

	# Activity time-scale sweep (uncomment/comment lines below)
	# (START: Activity time sweep)


	fil = activeFilament(dim = DIMS, Np = NP, radius = RADIUS, b0 = B0, k = K, S0 = S0, D0 = D0, bc = BC)
	
	activity_timescale = parameter # Activity time-scale (one compression and extension cycle)
	duty_cycle = 0.5	# Relative time for compression relative to total activity time-scale
	n_activity_cycles = 500 # No:of activity cycles we want to simulate
	Tf = activity_timescale*n_activity_cycles # Total simulation time
	time_step_save = 5 # This is roughly 4X the axial stretch time-scale which is the smallest time-scale in the dynamics
	Npts = int(Tf/time_step_save) # No:of time points saved

	# Activity parameters
	# Square-wave activity (Uncomment below)
	activity_parameters = {'type':'square-wave','activity time scale':activity_timescale, 'duty_cycle':duty_cycle, 
		'start phase':0}

	fil.simulate(Tf, Npts,  save = True, overwrite = False, 
		path = ROOT_PATH, sim_type = 'point', init_condition = {'shape':'line'}, 
	activity= activity_parameters)

	#( END : Activity time sweep)

#-------------------------------------------------------------
# Filament parameters
#-------------------------------------------------------------
NP = 32
D0 = 1.5
# Activity profile parameters
activity_timescale = 750 # Activity time-scale (one compression and extension cycle)
duty_cycle = 0.5	# Relative time for compression relative to total activity time-scale
n_activity_cycles = 500 # No:of activity cycles we want to simulate
Tf = activity_timescale*n_activity_cycles # Total simulation time

time_step_save = 5 # This is roughly 4X the axial stretch time-scale which is the smallest time-scale in the dynamics
Npts = int(Tf/time_step_save) # No:of time points saved

# Activity parameters
# Square-wave activity (Uncomment below)
activity_parameters = {'type':'square-wave','activity time scale':activity_timescale, 'duty_cycle':duty_cycle, 
	'start phase':0}

# Biphasic activity (Uncomment below)
# slow_timescale = 750
# fast_timescale = 250
# n_slow_cycles = 1
# n_fast_cycles = 3
# n_activity_cycles = 10 # No:of activity cycles we want to simulate (for biphasic activity this is the number of slow and fast cycles)
# Tf = n_activity_cycles*(n_slow_cycles*slow_timescale+n_fast_cycles*fast_timescale)
# time_step_save = 5 # This is roughly 4X the axial stretch time-scale which is the smallest time-scale in the dynamics
# Npts = int(Tf/time_step_save) # No:of time points saved
# activity_parameters = {'type':'biphasic','activity time scale':{'slow':slow_timescale, 'fast':fast_timescale}, 
#                        'duty_cycle':0.5,'N_cycles':{'slow':n_slow_cycles, 'fast':n_fast_cycles}, 
#                        'start_state':'slow', 'start phase':0}


# Normally distributed activity times
# activity_parameters={'type':'normal','activity time scale':activity_timescale, 
# 'n_cycles':n_activity_cycles, 'duty_cycle':duty_cycle, 'noise_scale':0.20}



# Activity strength sweep
# parameter_list = np.linspace(0.5,3, 10)
# parameter_list = np.linspace(0.25,0.75,10)
# parameter_list = np.linspace(0.15, 0.7, 15)

# Activity time sweep
parameter_list = np.arange(300, 800, 25, dtype = float)


# Number of initial conditions to simulate
num_initial_conditions = 2

parameter_list_full = []

for ii in range(num_initial_conditions):
	parameter_list_full.append(parameter_list)

parameter_list_full = np.array(parameter_list_full).flatten()

print(parameter_list_full)

num_cores = multiprocessing.cpu_count()

print("No:of detected cores: {}".format(num_cores))

num_cores = 12

results = Parallel(n_jobs=num_cores,  verbose=10)(delayed(run_parametric_simulation)(pid, parameter) for pid, parameter in enumerate(parameter_list_full))
