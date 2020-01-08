from Filament import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 

# Total simulation time
Tf = 100
# No:of time points saved
Npts = 1000

activityFreq = 1/2000

t_array = np.linspace(0, Tf+10, 500)

activity_profile = -signal.square(2*np.pi*activityFreq*t_array)

activity_Function =  interpolate.interp1d(t_array, activity_profile)

plt.figure()
plt.plot(t_array, activity_profile)
plt.show()

bc = {0:'clamped', -1:'free'}

fil = activeFilament(dim = 3, Np = 32, b0 = 4, k = 1, radius = 1, S0 = 0, D0 = 2, shape = 'sinusoid', bc =  activity_timescale = 1/activityFreq, simNotes = 'distActivity')

fil.plotFilament(r = fil.r0)






fil.simulate(Tf, Npts, activity_profile = activity_Function, save = True, overwrite = True)

# finalPos = fil.R[-1,:]

# fil.plotSimResult()

# fil.plotFilament(r = finalPos)

# fil.plotFlowFields(save = False)

# fil.plotFilament(r = finalPos)

# fil.plotFilamentStrain()

# fil.resultViewer()

fil.animateResult()