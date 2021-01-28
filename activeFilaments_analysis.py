import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
import os
import pyfilaments.analysisutils as analysis


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

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/2020-09-09/SimResults_Np_33_Shape_line_k_50_b0_2_F_0_S_0_D_1.5_scalefactor_500_1/SimResults_00.hdf5'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/2020-09-10/CompSearch_SimResults_Np_33_Shape_line_kappa_hat_5_k_20.0_b0_4_F_0_S_0_D_1.5_scalefactor_1000_1/SimResults_00.hdf5'

# file = '/home/deepak/LacryModelling_Local/ModellingResults/2021-01-26/SimResults_Np_65_Shape_line_kappa_hat_2.5_k_10_b0_2_F_0_S_0_D_1.5_activityTime_750_simType_point/SimResults_16.hdf5'

# file = '/home/deepak/LacryModelling_Local/ModellingResults/2021-01-22/ActivityTime_1000/SimResults_Np_33_Shape_line_kappa_hat_2.5_k_10_b0_2_F_0_S_0_D_1.5_activityTime_1000_simType_point/SimResults_00.hdf5'

file = '/home/deepak/LacryModelling_Local/ModellingResults/2021-01-23/ActivityTime_1000/SimResults_Np_33_Shape_line_kappa_hat_2.5_k_10_b0_2_F_0_S_0_D_1.5_activityTime_1000_simType_point/SimResults_00.hdf5'
# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/2020-09-09/LowSearchCoverage_SimResults_Np_33_Shape_line_k_50_b0_2_F_0_S_0_D_1.5_scalefactor_500_1/SimResults_00.hdf5'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/2020-09-07/GoodSearch_SimResults_Np_33_Shape_line_k_25_b0_2_F_0_S_0_D_1.5_scalefactor_500_1/SimResults_00.hdf5'

folder, *rest = os.path.split(file)

print(folder)

filament = analysis.analysisTools(file = file)

# filament.compute_bend_angles()

# alignParam = filament.compute_alignment_parameter()


# Calculate the filament length vs time

# filament.compute_arc_length()
# filament.plot_arclength_timeseries()

# # filament.plotFilament(r = filament.R[-1,:])

# filament.plot_tip_position()

# filament.filament_tip_coverage()

# filament.compute_arc_length()
# filament.compute_axial_bending_energy()

# filament.plot_timeseries(var = ['Filament arc length'])




# filament.compute_self_interaction_forces()


# # # # Plot the self-interaction forces vs time
# forces_x = filament.derived_data['self-interaction forces'][0:filament.Np, :]
# forces_y = filament.derived_data['self-interaction forces'][filament.Np:2*filament.Np, :]
# forces_z = filament.derived_data['self-interaction forces'][2*filament.Np:3*filament.Np, :]
# fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols = 3)
# c = ax0.pcolor(forces_x)
# ax0.set_title('Forces x')
# fig.colorbar(c, ax = ax0)
# c = ax1.pcolor(forces_y)
# ax1.set_title('Forces y')
# fig.colorbar(c, ax = ax1)
# c =ax2.pcolor(forces_z)
# ax2.set_title('Forces z')
# fig.colorbar(c, ax = ax2)
# plt.show()


filament.compute_tip_velocity()

plt.figure()
plt.plot(filament.Time[:-1], filament.derived_data['tip speed'], color = 'g', linewidth = 1)
plt.xlabel('Time')
plt.ylabel('Tip speed')
plt.show()


# filament.compute_head_orientation()
# filament.plot_timeseries(data = {'Tip cosine angle':[]})

# filament.plot_scatter(var_x = 'Filament arc length',var_y = 'Tip cosine angle', color_by = 'Time', save_folder = folder)


# filament.plot_phase_portrait(var_x = 'Axial energy', var_y = 'Bending energy', save_folder = None)

# filament.plot_phase_portrait(var_x = 'Filament arc length', var_y = 'Tip cosine angle', save_folder = folder)
# filament.plot_unique_tip_locations()

# filament.plot_coverage_vs_time()

# filament.plot_head_orientation_phase(save_folder = folder)

# filament.plot_energy_timeseries(save_folder = None)

# filament.plot_axial_vs_bending_energy(save_folder = folder)

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



