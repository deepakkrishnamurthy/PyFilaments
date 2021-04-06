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

# Activity profile parameters
activity_timescale = 750
duty_cycle = 0.5

# No:of activity cycles we want to simulate
n_activity_cycles = 1
# Total simulation time (for relaxation to eqbrm)
Tf = 1000

# activity_timescale = 1000

# Total simulation time
# No:of time points saved
time_step_save = 1
Npts = int(Tf/time_step_save)

# First simulate the relaxation to equilibrium using a clamped and fixed BC
bc = {0:'clamped', -1:'fixed'}
radius = 1
fil = activeFilament(dim = 3, Np = 32, radius = radius, b0 = 2.1*radius, 
	k = 20, S0 = 0, D0 = 0, bc = bc)

fil.simulate(Tf, Npts, save = True, overwrite = False, path = root_path, sim_type = 'point', 
	init_condition = {'shape':'line','angle':np.pi/4}, 
	activity={'type':'square-wave','activity_timescale':activity_timescale, 'duty_cycle':duty_cycle})


r_relaxed = fil.r
fil.plotFilament(r = r_relaxed, title = 'Relaxed filament shape')

bc = {0:'clamped', -1:'free'}
Tf = n_activity_cycles*activity_timescale
Npts = int(Tf/time_step_save)

fil = activeFilament(dim = 3, Np = 32, radius = radius, b0 = 2.1*radius, 
	k = 20, S0 = 0, D0 = 1.5, bc = bc)

fil.simulate(Tf, Npts, save = True, overwrite = False, path = root_path, sim_type = 'point', 
	init_condition = {'filament':r_relaxed}, 
	activity={'type':'square-wave','activity_timescale':activity_timescale, 'duty_cycle':duty_cycle})

fil.plotFilament(r = fil.r)
