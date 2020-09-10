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
# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_32_Shape_sinusoid_k_1_b0_4_S_0_D_-1_actTime_5000/SimResults_Np_32_Shape_sinusoid_k_1_b0_4_S_0_D_-1_actTime_5000.pkl'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2020-06-22/SimResults_Np_32_Shape_line_k_100_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/SimResults_Np_32_Shape_line_k_50_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_Np_32_Shape_line_k_50_b0_2_F_-1_S_0_D_0_actTime_0_scaleFactor_1_sedimentation.hdf5'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/SimResults_Np_32_Shape_line_k_100_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_Np_32_Shape_line_k_100_b0_2_F_-1_S_0_D_0_actTime_0_scaleFactor_1_sedimentation.hdf5'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/2020-07-09/SimResults_Np_33_Shape_sinusoid_k_50_b0_2_F_0_S_0_D_1.5_scalefactor_1000_1/SimResults_00.hdf5'
# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/2020-07-09/SimResults_Np_33_Shape_sinusoid_k_50_b0_2_F_0_S_0_D_1.5_scalefactor_1000_1ConstantDipole/SimResults_00.hdf5'
# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/2020-07-09/SimResults_Np_33_Shape_sinusoid_k_50_b0_2_F_0_S_0_D_1.5_scalefactor_1000_1ConstantDipole/SimResults_01.hdf5'
# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/2020-07-09/SimResults_Np_33_Shape_sinusoid_k_50_b0_2_F_0_S_0_D_1.5_scalefactor_1000_1/SimResults_01.hdf5'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/2020-09-07/SimResults_Np_33_Shape_line_k_25_b0_2_F_0_S_0_D_1.5_scalefactor_500_1/SimResults_01.hdf5'

file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/2020-09-09/SimResults_Np_33_Shape_line_k_50_b0_2_F_0_S_0_D_1.5_scalefactor_500_1/SimResults_00.hdf5'

filament = analysis.analysisTools(file = file)

# filament.compute_bend_angles()

# alignParam = filament.compute_alignment_parameter()


# Calculate the filament length vs time

filament.compute_arc_length()
filament.plot_arclength_timeseries()

# filament.plotFilament(r = filament.R[-1,:])

filament.plot_tip_position()

filament.filament_tip_coverage()




filament.plot_unique_tip_locations()

filament.plot_coverage_vs_time()



# # Plot euclidean distance between filaments vs time to check for convergence to steady state

# r_previous = filament.R[0, :]
# r_current = filament.R[0, :]

# distance_array = np.zeros(filament.Nt-1)
# r_com = np.zeros((filament.dim, filament.Nt))

# for ii in range(filament.Nt):

# 	r_previous = r_current

# 	r_current = filament.R[ii, :]

# 	r_com[:,ii] = [np.nanmean(r_current[:filament.Np-1]), 
# 		np.nanmean(r_current[filament.Np:2*filament.Np-1]), np.nanmean(r_current[2*filament.Np:3*filament.Np-1]) ] 

# 	if(ii!=0):  # Skip the first time step

# 		distance_array[ii-1] = filament.euclidean_distance(r_previous, r_current)


# print(distance_array)

# filament.plotFilament(r = r_current)

# # Plot pair Euclidean distance vs time

# plt.figure()
# plt.plot(filament.Time[:-1], distance_array, 'g',linestyle = '-')
# plt.xlabel('Time')
# plt.ylabel('Euclidean distance')
# plt.show()

# plt.figure()
# plt.plot(filament.Time, r_com[0,:], 'r',linestyle = '-')
# plt.plot(filament.Time, r_com[1,:], 'g',linestyle = '--')
# plt.plot(filament.Time, r_com[2,:], 'b',linestyle = ':')

# plt.xlabel('Time')
# plt.ylabel('COM position')
# plt.show()



