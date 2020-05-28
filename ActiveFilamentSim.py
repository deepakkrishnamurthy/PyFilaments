from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 

# Total simulation time
Tf = 500
# No:of time points saved
Npts = 500

activity_timescale = 2000
activityFreq = 1/activity_timescale

t_array = np.linspace(0, Tf+10, 500)

activity_profile = -signal.square(2*np.pi*activityFreq*t_array)

activity_Function =  interpolate.interp1d(t_array, activity_profile)

plt.figure()
plt.plot(t_array, activity_profile)
plt.show()

bc = {0:'clamped', -1:'free'}

fil = activeFilament(dim = 3, Np = 32, b0 = 4, k = 1, radius = 1, S0 = 0, D0 = 1.5, shape = 'sinusoid', bc = bc,  activity_timescale = activity_timescale)

fil.plotFilament(r = fil.r0)






fil.simulate(Tf, Npts, activity_profile = activity_Function, save = True, overwrite = True, path = '/Users/deepak/Dropbox/LacryModeling/ModellingResults')

# finalPos = fil.R[-1,:]

# fil.plotSimResult()

# fil.plotFilament(r = finalPos)

# fil.plotFlowFields(save = False)

# fil.plotFilament(r = finalPos)

# fil.plotFilamentStrain()

# fil.resultViewer()

# fil.animateResult()