# DEF file for defining constants and other static variables for SIMULATIONS
from sys import platform
import numpy as np

# Check which platform
if platform == "linux" or platform == "linux2":
	print("linux system")
	ROOT_PATH = '/home/deepak/ActiveFilamentsSearch_backup_3/ModellingResults'
	# root_path = '/home/deepak/Dropbox/LacryModeling/ModellingResults'
	

elif platform == 'darwin':
	print("OSX system")
	ROOT_PATH = '/Users/deepak/Dropbox/LacryModeling/'

# Naming conventions
FILE_NAME = 'sim_data' # Name for individual simulation files
FILE_FORMAT = '.hdf5'

# Filament parameter constants
DIMS = 3
RADIUS = 1
B0 = 2.1*RADIUS
BC = {0:'clamped', -1:'free'}
MU = 1/6.0  # Viscosity


N_IC = 10 # No:of filament ICs
ANGULAR_AMP_IC = np.pi/4 # Angular amplitude for generating filament ICs
TRANSVERSE_NOISE = 1E-6