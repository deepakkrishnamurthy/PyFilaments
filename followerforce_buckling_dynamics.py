'''Simulate buckling dynamics of filaments under tip follower-forces

	- Simulates filament dynamics from different orientation dynamics.
	- Computes filament orientation before and after one activity cycle (compressive-extensional) cycle

'''
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
	# root_path = '/home/deepak/LacryModelling_Local/SimulationData'
	root_path = '/home/deepak/Dropbox/LacryModeling/ModellingResults'
	

elif platform == 'darwin':
	print("OSX system")
	root_path = '/Users/deepak/Dropbox/LacryModeling/'


# Bond length
b0 = 4

# Activity profile parameters
duty_cycle = 0.5

# No:of activity cycles we want to simulate
n_activity_cycles = 1


init_angle_array = np.linspace(0, np.pi/2, 50)
bending_stiffness_array = [90, 95, 100]
# bending_stiffness_array = [15, 20, 25, 30, 35, 40, 50, 75, 100, 200]
activity_timescale_array = [2000]
# bending_stiffness_array = [15]
# activity_timescale_array = [750]



def run_parametric_simulation(pid, init_angle, stiffness, activity_timescale):
	radius = 1
	bc = {0:'clamped', -1:'free'}
	Tf = n_activity_cycles*activity_timescale
	time_step_save = 1
	Npts = int(Tf/time_step_save)

	fil = activeFilament(dim = 3, Np = 32, radius = radius, b0 = b0*radius, k = stiffness, S0 = 0, D0 = 1.5, bc = bc)

	print(init_angle)
	fil.simulate(Tf, Npts, save = True, overwrite = False, path = root_path, sim_type = 'point', 
	init_condition = {'shape':'line', 'angle': init_angle}, 
	activity={'type':'square-wave','activity_timescale':activity_timescale, 'duty_cycle':duty_cycle})


num_cores = multiprocessing.cpu_count()

num_cores = 12


for stiffness in bending_stiffness_array:
	for activity_timescale in activity_timescale_array:

		results = Parallel(n_jobs=num_cores,  verbose=10)(delayed(run_parametric_simulation)(pid, init_angle, stiffness, activity_timescale) for pid, init_angle in enumerate(init_angle_array))


# fil.plotFilament(r = fil.r)
