# Validation case 1: Sedimentation of as passive-filament:

from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 


# Filament parameters

Np = 32            	# number of particles
a = 1				# radius
b0 = 2*a 			# equilibrium bond length
k = 100				# Spring stiffness
mu = 1.0/6			# Fluid viscosity




# Total simulation time
Tf = 100
# No:of time points saved
Npts = 500

# Filament BC
bc = {0:'free', -1:'free'}

F_mag = -1			# Force on each particle. 


def elasto_gravitational_number():

	eg_number = k*a**2/(4*Np**3*abs(F_mag)*b0)

	return eg_number


tau_sedimentation = 6*np.pi*mu*a**2/abs(F_mag)

elasto_gravitational_length_scale = k*b0*a**2/(4*Np*abs(F_mag))

print('Elasto-gravitational number (beta): {}'.format(elasto_gravitational_number()))
print('Sedimentation time-scale: {}'.format(tau_sedimentation))

print('Elasto-gravitational length scale: {}'.format(elasto_gravitational_length_scale))

print('Spacing between colloids: {}'.format(b0))


filament = activeFilament(dim = 3, Np = Np, radius = a, b0 = b0, k = k, mu = mu,  F0 = F_mag, S0 = 0, D0 = 0, shape = 'line', bc = bc)

filament.plotFilament(r = filament.r0)



filament.simulate(Tf, Npts, sim_type = 'sedimentation', save = True, overwrite = True, path = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults')