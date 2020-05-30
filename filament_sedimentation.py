# Validation case 1: Sedimentation of as passive-filament:

from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 


# Filament parameters
a = 1
Np = 32            # radius and number of particles
b0 = 2*a 			# equilibrium bond length
k = 10				# Spring stiffness
mu = 1.0/6			# Fluid viscosity


# Total simulation time
Tf = 1000
# No:of time points saved
Npts = 500

# Filament BC
bc = {0:'free', -1:'free'}

F_mag = -1			# Force on the beads. 

filament = activeFilament(dim = 3, Np = Np, radius = a, b0 = b0, k = k, mu = mu,  F0 = F_mag, S0 = 0, D0 = 0, shape = 'line', bc = bc)

filament.plotFilament(r = filament.r0)



filament.simulate(Tf, Npts, sim_type = 'sedimentation', save = True, overwrite = True, path = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults')