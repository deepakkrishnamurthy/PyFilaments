# DEF file for defining constants and other static variables for ANALYSIS

from sys import platform
import numpy as np
import cmocean

# Check which platform
if platform == "linux" or platform == "linux2":
	print("linux system")
	ROOT_PATH = '/home/deepak/ActiveFilamentsSearch_backup_3/ModellingResults'
	# root_path = '/home/deepak/Dropbox/LacryModeling/ModellingResults'
	

elif platform == 'darwin':
	print("OSX system")
	ROOT_PATH = '/Users/deepak/Dropbox/LacryModeling/'


# Colors and Colormaps

COMP_COLOR = 'k'
EXT_COLOR = 'r'

SPATIAL_DENSITY_CMAP = cmocean.cm.deep

ACTIVITY_STRENGTH_CMAP = cmocean.cm.matter

ACTIVITY_TIME_CMAP = cmocean.cm.deep

# No:of cycles to ignore to remove transients
START_CYCLE = 100