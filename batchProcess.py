# Analysis batch processing
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
import os
import pyfilaments.analysisutils as analysis
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing

# Folder containing data
data_folder = '/home/deepak/LacryModelling_Local/SimulationData_ForAnalysis/2021-02-07'

# Find all simulation data files and create a list
files_list = []
 # Walk through the folders and identify the simulation data files
for dirs, subdirs, files in os.walk(data_folder, topdown=False):
   
	root, subFolderName = os.path.split(dirs)
	  
	for fileNames in files:
		if('.hdf5' in fileNames):
			
			files_list.append(os.path.join(dirs,fileNames))


# for file in tqdm(files_list):
	
def run_filament_analysis(file):
	print('Analyzing file ...')
	print(file)

	filament = analysis.analysisTools(file = file)
	filament.filament_tip_coverage(save = True)


num_cores = multiprocessing.cpu_count()

num_cores = 12

results = Parallel(n_jobs=num_cores,  verbose=10)(delayed(run_filament_analysis)(file) for file in tqdm(files_list))
