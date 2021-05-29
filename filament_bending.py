# Validation case 2: Bending of a clamped filament subject to a transverse force at the tip.
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
from sys import platform
from joblib import Parallel, delayed
import multiprocessing

from pyfilaments.activeFilaments import activeFilament


# Filament parameters
Np = 64            	# number of particles
a = 1				# radius
b0 = 2.1*a 			# equilibrium bond length
k_array = [500]				# Spring stiffness
mu = 1.0/6			# Fluid viscosity


# Total simulation time
Tf = 100
# No:of time points saved
Npts = 500

activity_timescale = Tf
duty_cycle = 0

# Filament BC
bc = {0:'clamped', -1:'free'}

F_mag = -1			# Force on distal particle.

def run_parametric_sweep(pid, k):
	
	filament = activeFilament(dim = 3, Np = Np, radius = a, b0 = b0, k = k, mu = mu,  F0 = F_mag, S0 = 0, D0 = 0, bc = bc)

	filament.plotFilament(r = filament.r0)

	# Check which platform to define the OpenMP flags
	if platform == "linux" or platform == "linux2":
		print("linux system")
		root_path = '/home/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults'
		

	elif platform == 'darwin':
		print("OSX system")
		root_path = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults'

	filament.simulate(Tf, Npts, sim_type = 'cantilever', save = True, overwrite = False, 
		path = root_path, pid = pid, activity={'type':'square-wave','activity_timescale':activity_timescale, 'duty_cycle':duty_cycle})


for ii, k in enumerate(k_array):
	run_parametric_sweep(ii, k)

# num_cores = multiprocessing.cpu_count()
# results = Parallel(n_jobs=num_cores)(delayed(run_parametric_sweep)(pid, parameter) for pid, parameter in enumerate(k_array)) 