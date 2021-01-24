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


activity_timescale = 1000
activityFreq = 1.0/activity_timescale


# Total simulation time
Tf = activity_timescale*500


# activity_timescale = 1000
activityFreq = 1.0/activity_timescale

print('Activity frequency: {}'.format(activityFreq))

# Total simulation time

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

fil = activeFilament(dim = 3, Np = 64, radius = 1, b0 = 2, k = 20, S0 = 0, D0 = 1.5, bc = bc)


fil.plotFilament(r = fil.r0)


fil.simulate(Tf, Npts, activity_profile = activity_Function, save = True, overwrite = False, path = root_path ,
  activity_timescale = activity_timescale, sim_type = 'point', init_condition = {'shape':'line'})

