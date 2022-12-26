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

# List of filament parameters that can be varied for parametric sweeps
PARAMETERS = ['Np', 'radius', 'b0', 'k' , 'mu' , 'F0', 'S0', 'D0', 
					 'scale_factor','bending_axial_scalefactor','bc']

# Filament parameter constants and default values
DIMS = 3
RADIUS = 1
B0 = 2.1*RADIUS
S0 = 0
D0 = 1.5
BC = {0:'clamped', -1:'free'}
MU = 1/6.0  # Viscosity
K = 25 # Axial stiffness

INIT_SHAPE = {'shape':'line'}

N_IC = 10 # No:of filament ICs
ANGULAR_AMP_IC = np.pi/12 # Angular amplitude for generating filament ICs
TRANSVERSE_NOISE = 1E-6

# Spatial activity distributions
SPATIAL_ACTIVITY_TYPES = ['point', 'dist', 'lacry']

# Temporal activity types
TEMPORAL_ACTIVITY_TYPES = ['square-wave', 'biphasic', 'normal','lognormal']

BENDING_AXIAL_SCALEFACTOR = 0.25 # 1/4 for a homogeneous elastic rod

# Scale-factor (between strength of head cilia and neck cilia in Lacry)
SCALE_FACTOR = 2 # Arbitrary for now


# Scale factor between median extension durations and compression durations (from experimental measurements)
EXT_COMP_SCALEFACTOR = 1.5

SIGMA_EXT = 0.89 # From fitting of experimental data to a lognormal distribution
SIGMA_COMP = 0.86 # From fitting of experimental data to a lognormal distribution