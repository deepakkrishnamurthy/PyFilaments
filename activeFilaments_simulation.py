from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
from sys import platform
import pandas as pd
import time

# Check which platform
if platform == "linux" or platform == "linux2":
	print("linux system")
	root_path = '/home/deepak/Dropbox/LacryModeling/ModellingResults'
	

elif platform == 'darwin':
	print("OSX system")
	root_path = '/Users/deepak/Dropbox/LacryModeling/ModellingResults'

# Activity profile parameters
activity_timescale = 750
duty_cycle = 0.5

# No:of activity cycles we want to simulate
n_activity_cycles = 5
# Total simulation time
Tf = activity_timescale*n_activity_cycles

# activity_timescale = 1000

# Total simulation time
# No:of time points saved
time_step_save = 10
Npts = int(Tf/time_step_save)

bc = {0:'clamped', -1:'free'}


fil = activeFilament(dim = 3, Np = 32, radius = 1, b0 = 2.1, k = 40, F0 = 0, S0 = 0, D0 = 1.5, bc = bc, clamping_vector = [1,0,0])


fil.plotFilament(r = fil.r0)


fil.simulate(Tf, Npts, save = True, overwrite = False, path = root_path, sim_type = 'point', 
	init_condition = {'shape':'line'}, 
	activity={'type':'square-wave','activity_timescale':activity_timescale, 'duty_cycle':duty_cycle})


fil.plotFilament(r = fil.r)
