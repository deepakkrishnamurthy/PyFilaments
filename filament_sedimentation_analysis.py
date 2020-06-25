# bench-marking simulations analysis
# Compares simulated steady-state shapes of filaments with predictions from theory.

from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
import pyfilaments.analysisutils as analysis

file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/SimResults_Np_32_Shape_line_k_100_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_Np_32_Shape_line_k_100_b0_2_F_-1_S_0_D_0_actTime_0_scaleFactor_1_sedimentation.hdf5'

filament = analysis.analysisTools(file = file)