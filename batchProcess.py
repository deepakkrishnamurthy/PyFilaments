# Analysis batch processing
import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
import os
import pyfilaments.analysisutils as analysis
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing

# Folder containing data
# data_folder = '/home/deepak/LacryModelling_Local/SimulationData_ForAnalysis/2021-02-05'
# data_folder = '/home/deepak/LacryModelling_Local/SimulationData/2021-02-09'
# data_folder = '/Volumes/DEEPAK-1TB/ActiveFilaments_Simulations_Backup/BendingStiffnessSweeps/b0_4_activity_time_2000/2021-02-27'
# data_folder = '/media/deepak/DEEPAK-1TB/ActiveFilaments_Simulations_Backup/BendingStiffnessSweeps/b0_4_activity_time_2000/2021-02-27'
# data_folder = '/media/deepak/DEEPAK-1TB/ActiveFilaments_Simulations_Backup/BendingStiffnessSweeps/AnalysisData'
# data_folder = '/home/deepak/ActiveFilamentsSearch_backup_3/BendingStiffnessSweeps/analysis_test'
data_folder = '/home/deepak/ActiveFilamentsSearch_backup_3/ModellingResults'

print(os.listdir(data_folder))

# Find all simulation data files and create a list
files_list = []
 # Walk through the folders and identify the simulation data files
for dirs, subdirs, files in os.walk(data_folder, topdown=False):
	
	root, subFolderName = os.path.split(dirs)

	for fileNames in files:
		if(fileNames.endswith('hdf5') and fileNames[0] != '.'):
			files_list.append(os.path.join(dirs,fileNames))

print('Simulation files: ', files_list)

	
def run_filament_analysis(file):
	print('Analyzing file ...')
	print(file)

	filament = analysis.analysisTools(file = file)


	root_folder, file_name = os.path.split(file)

	

	# Create a data-packet to save for further analysis
	# Collect simulation metadata
	metadata_path = os.path.join(root_folder, 'metadata.csv')
	assert(os.path.exists(metadata_path))
	df = pd.read_csv(metadata_path)

	

	# Check if simulation was completed or terminated before completion
	if(int(filament.Time[-1]) == int(df['Simulation time'])):
		simulation_completed = True
		filament.filament_tip_coverage(save = True)
		# Classify the dynamics
		periodic_flag, min_period = filament.classify_filament_dynamics()

		df['periodic dynamics'] = periodic_flag
		df['period'] = min_period
		df['max unique locations'] = filament.derived_data['unique position count'][-1]
	
	else:
		simulation_completed = False

	df['simulation completed'] = simulation_completed

	save_file = file_name[:-5] + '_analysis.csv'
	save_folder = os.path.join(root_folder, 'Analysis')

	if(not os.path.exists(save_folder)):
		os.makedirs(save_folder)
	df.to_csv(os.path.join(save_folder, save_file))

	# filament.plot_filament_centerlines(stride = 100, save = True, color_by = 'Time')
	# filament.plot_tip_scatter_density(save = True)

# for file in tqdm(files_list):
# 	run_filament_analysis(file)

num_cores = multiprocessing.cpu_count()

num_cores = 12

results = Parallel(n_jobs=num_cores,  verbose=10)(delayed(run_filament_analysis)(file) for file in tqdm(files_list))
