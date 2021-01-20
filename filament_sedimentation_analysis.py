# bench-marking simulations analysis
# Compares simulated steady-state shapes of filaments with predictions from theory.

from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
import pyfilaments.analysisutils as analysis
import pandas as pd
import seaborn as sns


from matplotlib import rcParams
from matplotlib import rc
#rcParams['axes.titlepad'] = 20 
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
### for Palatino and other serif fonts use:
##rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=False)
#plt.rc('font', family='serif')

rc('font', family='sans-serif') 
rc('font', serif='Helvetica') 
rc('text', usetex='false') 
rcParams.update({'font.size': 12})
# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/SimResults_Np_32_Shape_line_k_100_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_Np_32_Shape_line_k_100_b0_2_F_-1_S_0_D_0_actTime_0_scaleFactor_1_sedimentation.hdf5'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2020-06-23/SimResults_Np_32_Shape_line_k_2000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2020-06-24/SimResults_Np_32_Shape_line_k_5000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2020-06-24/SimResults_Np_33_Shape_line_k_5000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'

# file = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2020-06-25/SimResults_Np_33_Shape_line_k_10000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'


# Plot data with different filament stiffness on top of each other

# file_1 = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2020-06-25/SimResults_Np_33_Shape_line_k_2000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
# file_2 = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2020-06-24/SimResults_Np_33_Shape_line_k_5000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
# file_3 = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2020-06-25/SimResults_Np_33_Shape_line_k_10000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'

# N = 64 filaments
# file_1 = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2020-06-24/SimResults_Np_64_Shape_line_k_500_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
# file_2 = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2020-06-24/SimResults_Np_64_Shape_line_k_1000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
# file_3 = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2020-06-24/SimResults_Np_64_Shape_line_k_2000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
# file_4 = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2020-06-24/SimResults_Np_64_Shape_line_k_5000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'


# file_1 = '/home/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2021-01-18/SimResults_Np_64_Shape_line_k_1000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
# file_2 = '/home/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2021-01-18/SimResults_Np_64_Shape_line_k_2000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
# file_3 = '/home/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2021-01-18/SimResults_Np_64_Shape_line_k_5000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
# file_4 = '/home/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2021-01-18/SimResults_Np_64_Shape_line_k_10000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'

# file_1 = '/home/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2021-01-19/SimResults_Np_33_Shape_line_k_5000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
# file_2 = '/home/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2021-01-19/SimResults_Np_33_Shape_line_k_10000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
# file_3 = '/home/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2021-01-19/SimResults_Np_33_Shape_line_k_20000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'

file_1 = '/home/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2021-01-19/SimResults_Np_48_Shape_line_k_10000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
file_2 = '/home/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2021-01-19/SimResults_Np_48_Shape_line_k_20000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'
file_3 = '/home/deepak/Dropbox/LacryModeling/ModellingResults/BenchmarkingResults/2021-01-19/SimResults_Np_48_Shape_line_k_50000_b0_2_F_-1_S_0_D_0_scalefactor_0_1/SimResults_00.hdf5'

symbols = ['o', 's', '*', 'd']
colors = ['r', 'g', 'b', 'm']

files = [file_1, file_2, file_3]

print(len(files))

for index, file in enumerate(files):
	filament = analysis.analysisTools(file = file)


	com_array = np.zeros(filament.Nt)
	# Extract the steady-state sedimentation velocity of the filament
	for ii in range(filament.Nt):

		r = filament.R[ii,:]

		r_com = filament.filament_com(r)

		# We only want the y-direction
		com_array[ii] = r_com[1]
		
	U_array = np.diff(com_array)/np.diff(filament.Time)

	# Plot sedimentation velocity vs time
	fig1 = plt.figure(1)
	plt.plot(filament.Time[:-1:5], U_array[::5], color = colors[index], marker= symbols[index], label = 'k={}'.format(filament.k), alpha = 0.75)
	plt.xlabel('Time')
	plt.ylabel('Sedimentation speed')
	plt.title('Sedimentation speed vs Time')
	plt.legend(fontsize=14)
	if(index==len(files)-1):
		plt.savefig('SedimentationSpeed.svg', dpi = 300)
		plt.savefig('SedimentationSpeed.png', dpi = 300)
	# plt.show(block = False)


	U_sedimentation = U_array[-1]  # The convergence criterion for the simulations ensures this value has reached steady-state

	# Filament half-length
	l = (filament.Np-1)*filament.b0/2

	# Elastic modulus
	E = filament.k*filament.b0/(np.pi*filament.radius**2)

	# Area moment of inertia
	I = np.pi*(filament.radius**4)/4

	aspect_ratio = l/(filament.radius)

	print(50*'*')
	print('Filament elastic modulus: {}'.format(E))

	print('Filament sedimentation speed: {}'.format(U_sedimentation))
	print('Filament half-length: {}'.format(l))

	# Non-dimensional scaling factor D = 2\pi \mu U l^4/(E I \ln(\kappa)^2),
	D = 2*np.pi*filament.mu*abs(U_sedimentation)*(l**4)/(E*I*np.log(aspect_ratio)**2)

	print('Scaling factor (D): {}'.format(D))
	print(50*'*')

	def elasto_gravitational_number():

		eg_number = filament.k*filament.radius**2/(4*filament.Np**3*abs(filament.F0)*filament.b0)

		return eg_number

	print(50*'*')
	print('Elastogravitational number (beta): {}'.format(elasto_gravitational_number()))
	print(50*'*')
	# def filament_shape_theory(x):
	# 	y = -(1/24)*(((1+x)**4)*np.log(1+x) + ((1 - x)**4)*np.log(1 - x) - (3/16 + 2*np.log(2))*x**4 - (1 + 12*np.log(2))*x**2)

	filament_shape_theory = lambda x: -(1/24)*(((1 + x)**4)*np.log(1+x) + ((1 - x)**4)*np.log(1 - x) - (3/16 + 2*np.log(2))*(x**4) - (1 + 12*np.log(2))*(x**2))

	x_filament = filament.R[-1, :filament.Np] - np.nanmean(filament.R[-1, :filament.Np]) 
	y_filament = filament.R[-1, filament.Np: 2*filament.Np] - np.min(filament.R[-1, filament.Np: 2*filament.Np])


	print(np.min(y_filament))
	print(np.nanmean(x_filament))

	theory_results = '/home/deepak/Dropbox/LacryModeling/AnalysisResults/BenchmarkingSimulations/Xu_and_Nadim_Figure1.csv'

	df = pd.read_csv(theory_results)



	fig2 = plt.figure(2)
	plt.scatter(x_filament, y_filament, 50, marker = symbols[index], color=colors[index],label = 'k={}'.format(filament.k), alpha = 0.75)
	plt.title('Steady state shape (non scaled)')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend(fontsize=10)
	# plt.axis('equal')
	# plt.show(block = False)
	if(index==len(files)-1):
		plt.savefig('SteadyStateShape_raw_unequal.svg', dpi = 300)
		plt.savefig('SteadyStateShape_raw_unequal.png', dpi = 300)

	fig3 = plt.figure(3)
	plt.scatter(x_filament/l, y_filament/D, 50, marker = symbols[index], color=colors[index], label = 'Simulation '+ 'k={}'.format(filament.k), alpha = 0.75)
	if(index==0):
		plt.plot(df['x'], df['y'], 'k-', label = "Theory")
	plt.title('Steady state shape (scaled)')
	plt.xlabel('x/l')
	plt.ylabel('y/D')
	plt.legend(fontsize=10)

	if(index==len(files)-1):
		plt.savefig('SteadyStateShape_scaled.svg', dpi = 300)
		plt.savefig('SteadyStateShape_scaled.png', dpi = 300)
	# plt.show(block = False)


plt.show()


