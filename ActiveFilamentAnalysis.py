from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
import pyfilaments.analysisutils as analysis


# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_33_Shape_sinusoid_k_3_b0_4_S_0_D_1_actTime_2500_distActivity_contract_extension_sf_4/SimResults_Np_33_Shape_sinusoid_k_3_b0_4_S_0_D_1_actTime_2500.pkl'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_33_Shape_sinusoid_k_1_b0_4_S_0_D_0_actTime_2000_scalefactor_1_point/SimResults_Np_33_Shape_sinusoid_k_1_b0_4_S_0_D_0_actTime_2000_scaleFactor_1_point.pkl'

file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_33_Shape_sinusoid_k_1_b0_4_S_0_D_1.5_actTime_2000_scalefactor_1_point/SimResults_Np_33_Shape_sinusoid_k_1_b0_4_S_0_D_1.5_actTime_2000_scaleFactor_1_point.hdf5'

filament = analysis.analysisTools(file = file)

filament.BendAngle()

alignParam = filament.alignmentParameter()

print(alignParam)