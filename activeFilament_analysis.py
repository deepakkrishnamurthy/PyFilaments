from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
import pyfilaments.analysisutils as analysis

# import tkinter
# from tkinter import filedialog

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_33_Shape_sinusoid_k_3_b0_4_S_0_D_1_actTime_2500_distActivity_contract_extension_sf_4/SimResults_Np_33_Shape_sinusoid_k_3_b0_4_S_0_D_1_actTime_2500.pkl'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_33_Shape_sinusoid_k_1_b0_4_S_0_D_0_actTime_2000_scalefactor_1_point/SimResults_Np_33_Shape_sinusoid_k_1_b0_4_S_0_D_0_actTime_2000_scaleFactor_1_point.pkl'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_33_Shape_sinusoid_k_1_b0_4_S_0_D_1.5_actTime_2000_scalefactor_1_point/SimResults_Np_33_Shape_sinusoid_k_1_b0_4_S_0_D_1.5_actTime_2000_scaleFactor_1_point.hdf5'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/2020-06-18/SimResults_Np_33_Shape_sinusoid_k_20_b0_4_F_0_S_0_D_1.5_scalefactor_1000_1/SimResults_00.hdf5'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_32_Shape_sinusoid_k_1_b0_4_S_0_D_-1_actTime_1000.0/SimResults_Np_32_Shape_sinusoid_k_1_b0_4_S_0_D_-1_actTime_1000.pkl'
# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_32_Shape_sinusoid_k_1_b0_4_S_0_D_-1_actTime_2000/SimResults_Np_32_Shape_sinusoid_k_1_b0_4_S_0_D_-1_actTime_2000.pkl'
# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_32_Shape_sinusoid_k_1_b0_4_S_0_D_-1_actTime_3000/SimResults_Np_32_Shape_sinusoid_k_1_b0_4_S_0_D_-1_actTime_3000.pkl'
file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_32_Shape_sinusoid_k_1_b0_4_S_0_D_-1_actTime_5000/SimResults_Np_32_Shape_sinusoid_k_1_b0_4_S_0_D_-1_actTime_5000.pkl'
# root = tkinter.Tk()
# root.withdraw()
# file  = filedialog.askopenfilename(parent=root,initialdir="/",title='Please select a directory')

filament = analysis.analysisTools(file = file)

# filament.compute_bend_angles()

# alignParam = filament.compute_alignment_parameter()


# Calculate the filament length vs time

# filament.compute_arc_length()


# filament.plot_arclength_timeseries()

filament.plotFilament(r = filament.R[-1,:])

filament.plot_tip_position()