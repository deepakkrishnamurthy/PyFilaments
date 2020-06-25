from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 

# Total simulation time
Tf = 1000
# No:of time points saved
Npts = 500

activity_timescale = 200
activityFreq = 1/activity_timescale

t_array = np.linspace(0, Tf+10, 500)

activity_profile = -signal.square(2*np.pi*activityFreq*t_array)

activity_Function =  interpolate.interp1d(t_array, activity_profile)

plt.style.use('dark_background')
plt.figure()
plt.plot(t_array, activity_profile)
plt.show()

bc = {0:'clamped', -1:'free'}

fil = activeFilament(dim = 3, Np = 32, b0 = 4, k = 20, radius = 1, S0 = 0, D0 = 1.5, shape = 'sinusoid', bc = bc)

fil.plotFilament(r = fil.r0)




# Check which platform
if platform == "linux" or platform == "linux2":
	print("linux system")
	root_path = '/home/deepak/Dropbox/LacryModeling/ModellingResults'
	

elif platform == 'darwin':
	print("OSX system")
	root_path = '/Users/deepak/Dropbox/LacryModeling/ModellingResults'
	

# elif platform == "win32":
# 	print("Windows")
# 	with open(os.devnull, 'w') as fnull:
# 		exit_code = subprocess.call([compiler, '-Xpreprocessor', '-fopenmp', '-lomp', filename], stdout=fnull, stderr=fnull)


fil.simulate(Tf, Npts, activity_profile = activity_Function, save = True, overwrite = False, path = root_path ,
			activity_timescale = activity_timescale, sim_type = 'point')

# finalPos = fil.R[-1,:]

# fil.plotSimResult()

# fil.plotFilament(r = finalPos)

# fil.plotFlowFields(save = False)

# fil.plotFilament(r = finalPos)

# fil.plotFilamentStrain()

# fil.resultViewer()

# fil.animateResult()