from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
from sys import platform

from joblib import Parallel, delayed
import multiprocessing

# Check which platform
if platform == "linux" or platform == "linux2":
	print("linux system")
	root_path = '/home/deepak/LacryModelling_Local/SimulationData'
	

elif platform == 'darwin':
	print("OSX system")
	root_path = '/Users/deepak/Dropbox/LacryModeling/'

activity_timescale = 2000
activityFreq = 1.0/activity_timescale

print('Activity frequency: {}'.format(activityFreq))
# Total simulation time
Tf = activity_timescale*500
print('Total simulation time: {}'.format(Tf))
# No:of time points saved
Npts = int(Tf/10)
t_array = np.linspace(0, Tf+10, Npts)

activity_profile = -signal.square(2*np.pi*activityFreq*t_array)
activity_Function =  interpolate.interp1d(t_array, activity_profile)

plt.style.use('dark_background')
plt.figure()
plt.plot(t_array, activity_profile)
plt.show()

bc = {0:'clamped', -1:'free'}

def run_parametric_simulation(pid, parameter):
	radius = 1
	fil = activeFilament(dim = 3, Np = 32, radius = radius, b0 = 4*radius, k = parameter, S0 = 0, D0 = 1.5, bc = bc)

	fil.simulate(Tf, Npts, activity_profile = activity_Function, save = True, overwrite = False, path = root_path ,
			activity_timescale = activity_timescale, sim_type = 'point', init_condition = {'shape':'line'}, pid = pid)


parameter_list = np.array([65, 75])
num_initial_conditions = 3

parameter_list_full = []

for ii in range(num_initial_conditions):
	parameter_list_full.append(parameter_list)

parameter_list_full = np.array(parameter_list_full).flatten()

print(parameter_list_full)

num_cores = multiprocessing.cpu_count()

num_cores = 12

results = Parallel(n_jobs=num_cores,  verbose=10)(delayed(run_parametric_simulation)(pid, parameter) for pid, parameter in enumerate(parameter_list_full))