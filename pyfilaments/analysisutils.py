# Utility functions for analyzing filament shapes and dynamics
import sys
if 'init_modules' in globals().keys():
	# second or subsequent run: remove all but initially loaded modules
	for m in sys.modules.keys():
		if m not in init_modules:
			del(sys.modules[m])
else:
	# first run: find out which modules were initially loaded
	init_modules = sys.modules.keys()
	print(init_modules)

import os
import numpy as np
import imp
from pyfilaments.activeFilaments import activeFilament
from pyfilaments._def_analysis import *
import filament.filament as filament_operations
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy import interpolate
from scipy.stats import gaussian_kde
import seaborn as sns
import h5py

import cmocean
# Figure parameters
from matplotlib import rcParams
from matplotlib import rc
from matplotlib import cm
from tqdm import tqdm



rc('font', family='sans-serif') 
rc('font', serif='Helvetica') 
rc('text', usetex='false') 
rcParams.update({'font.size': 12})

class analysisTools(activeFilament):
	"""	Analysis class for active filaments. 
		
		Can be initialized in two ways.
		1. Pass an existing active filaments object.
		2. Pass a file containing simulation data.
	"""
	def __init__(self, filament = None, file = None):
		
		# Initialize the parent activeFilament class so its attr are available
		super().__init__()

		# Dict to store derived datasets
		self.derived_data = {'Filament arc length':[], 'Tip unit vector': [],
		'Axial energy':[],'Bending energy':[], 'common time array':{'compression':[], 'extension':[]},
		"Tip angle":{'compression':[], 'extension':[]}, 
		'Base-Tip angle':{'compression':[], 'extension':[]},
		'Tip decorrelation':{'compression':[], 'extension':[]},
		'Base-Tip decorrelation':{'compression':[], 'extension':[]},
		'constant phase':{'compression':[], 'extension':[]}, 
		'start end indices':{'compression':[], 'extension':[]}, 
		'Tip reorientation':{'compression':[], 'extension':[]},
		'Base-Tip reorientation':{'compression':[], 'extension':[]}}

		# Set the attributes to those of the filament on which we are doing analysis. 
		if(filament is not None):
			fil_attr = filament.__dict__
			for key in fil_attr.keys():
				setattr(self, key, getattr(filament, key))
			# Reload the cython sub-modules using the loaded parameters
			self.filament_operations = filament_operations.filament_operations(self.Np, self.dim, self.radius, self.b0, self.k, 
				self.kappa_hat_array, ljrmin = 2.1*self.radius, ljeps = 0.01)
		# If a saved file is supplied, then load it into memory.
		elif(file is not None):
			self.load_data(file)
			# Reload the cython sub-modules using the loaded parameters
			self.filament_operations = filament_operations.filament_operations(self.Np, self.dim, self.radius, self.b0, self.k, 
				self.kappa_hat_array, ljrmin = 2.1*self.radius, ljeps = 0.01)

			self.find_num_time_points()

			# Find time points corresponding to activity phase =0 and activity phase = pi
			self.derived_data['Phase'] = 2*np.pi*(self.Time%self.activity_timescale)/self.activity_timescale
			self.derived_data['head pos x'] = self.R[:, self.Np-1]
			self.derived_data['head pos y'] = self.R[:, 2*self.Np-1]
			self.derived_data['head pos z'] = self.R[:, 3*self.Np-1]

			# Get the root path for the saved data
			self.rootFolder, self.dataName = os.path.split(file)

			*rest,self.dataFolder = os.path.split(self.rootFolder)
			# Sub-folder in which to save analysis data and plots
			self.subFolder = 'Analysis'

			# Create a sub-folder to save Analysis results
			self.analysisFolder = os.path.join(self.rootFolder, self.subFolder)
			self.allocate_variables()

			# Load metadata
			self.df_metadata = pd.read_csv(os.path.join(self.rootFolder, 'metadata.csv'))

			# Activity cycles per simulation (only exact for deterministic activity profiles)
			self.activity_cycles = int(self.Time[-1]/self.activity_timescale) # Number of activity cycles

			self.compute_scales()

	def allocate_variables(self):
		self.tangent_angles_matrix = None
		self.covariance_matrix = None
		self.eigenvectors_sig = None
		self.d_sig = None

	def create_analysis_folder(self):
		"""	Create a sub-folder to store analysis data and plots
		"""
		if(not os.path.exists(self.analysisFolder)):
			os.makedirs(self.analysisFolder) 
		
	def relaxation_time_scales(self):
		"""	Calculate relaxation time-scales for the active filament.
		"""
		self.L = (self.Np - 1)*self.b0
		self.kappa = self.kappa_hat*self.b0
		self.tau_stretch = self.mu*self.L/self.k
		self.tau_bend = self.mu*self.L**4/(self.kappa)
		if(self.D0 != 0):
			self.tau_swim = 1/(self.radius*self.D0)	# Time-scale for an active colloid to swim its own length.
		else:
			self.tau_swim = np.nan
		
		print(50*'*')
		print('Time-scales')
		print(50*'*')
		print('Stretch relzation time: {}'.format(round(self.tau_stretch, 2)))
		print('Bend relaxation time: {}'.format(round(self.tau_bend, 2)))
		print('Active motility time-scale: {}'.format(round(self.tau_swim, 2)))


	def compute_scales(self):

		self.length_scale = self.radius
		self.time_scale = (self.radius*self.D0)**(-1)

		self.velocity_scale = self.length_scale/self.time_scale

		self.force_scale = self.mu*self.D0*(self.radius**3)

	def compute_dimensionless_groups(self):
		"""	Calculate derived dimensionles groups based on the intrinisic parameters. 
		"""

		self.f_a = self.mu*self.D0*(self.radius**3)/self.L
		self.activity_number = (self.mu*self.radius**3*self.L**2*self.D0/self.kappa)

		print(50*'*')
		print('Dimensionless numbers')
		print(50*'*')
		print('Force per unit lenth due to activity: {}'.format(round(self.f_a, 5)))
		print('Activity number: {}'.format(round(self.activity_number, 5)))
		print(50*'*')

   

	def dotProduct(self, a, b):
		""" Vector dot product of arrays of vectors (eg. nDims x N) where the first axis is the dimensionality of the vector-space.
			Returns: 
			shape 1 x N
		"""
		c = np.sum(a*b,axis=0)
		return  c

	def find_num_time_points(self):

		self.Nt, *rest  = np.shape(self.R)
		self.avg_time_step = np.max(self.Time)/self.Nt
		# print('Time step: {}'.format(np.max(self.Time)/self.Nt))
			
	def filament_com(self, r):

		r_com = np.zeros(self.dim)

		for ii in range(self.dim):

			r_com[ii] = np.nanmean(r[ii*self.Np: (ii+1)*self.Np-1])

		# r_com = [np.nanmean(r[:self.Np-1]), 
		# np.nanmean(r[self.Np:2*self.Np-1]), np.nanmean(r[2*self.Np:3*self.Np-1]) ] 

		return r_com

	def compute_axial_strain(self, R = None):

		if(R is not None):
			strain_vector = np.zeros((self.Nt, self.Np - 1))
			for ii in range(self.Nt):

				# Set the current filament positions to those from the simulation result at time point ii

				self.r = R[ii,:]

				self.get_separation_vectors()
				strain_vector[ii,:] = link_distance/self.b0

		# Size (Nt, self.Np - 1)
		return strain_vector

	def compute_bond_angles(self):
		"""	Compute the angle between adjacent bonds in the filament/polymer.
		"""
		self.cosAngle = np.zeros_like(self.R)

		for ii in range(self.Nt):

			self.r = self.R[ii, :]
			self.get_separation_vectors()



	def compute_tangent_angles(self):
		""" Calculate the local tangent angle of the filament relative to the unit cartesian axis fixed to the lab reference frame. 
			Assumes tangent vectors have already been computed for the current filament shape.
		"""
		# self.tangent_angles = np.zeros_like(self.R) # First calculate tangent angles at the sphere locations.
		self.get_tangent_vectors()
		tangent_angles = np.arctan2(self.t_hat[1,:], self.t_hat[0,:])

		return tangent_angles

	def compute_tangent_angle_matrix(self):
		""" Compute the tangent angles of the filament both over length (columns) and time (rows)
		"""
		n_points = 100 # No:of points used for interpolating the tangent angle representation of the filament shape.
		particles_array = np.linspace(0, 1, self.Np)
		points_array = np.linspace(0, 1, n_points)
		
		n_time = int(self.Nt)
		self.tangent_angles_matrix = np.zeros((n_time, n_points))

		for ii in range(n_time):
			self.r = self.R[ii, :]
			self.get_separation_vectors()
			tangent_angles = self.compute_tangent_angles()
			tangent_angles_fun = interpolate.interp1d(particles_array, tangent_angles, kind = 'linear')

			self.tangent_angles_matrix[ii, :] = tangent_angles_fun(points_array)

	def compute_shape_covariance_matrix(self):
		""" Calculates the shape covariance matrix for a list of filament shapes.

		Rows: time points
		Columns: Arc length 
	
		"""
		self.phi_0 = np.nanmean(self.tangent_angles_matrix, axis = 0)

		n_times, n_points = np.shape(self.tangent_angles_matrix)
		print('No:of spatial points: {}'.format(n_points))
		print('No:of time points: {}'.format(n_times))
		print(np.shape(np.tile(self.phi_0, (n_times, 1))))
		self.variance_matrix = self.tangent_angles_matrix - np.tile(self.phi_0, (n_times, 1))
		print(np.shape(self.variance_matrix))
		assert(np.shape(self.variance_matrix) == (n_times, n_points))
		self.covariance_matrix = np.matmul(self.variance_matrix.T, self.variance_matrix)

	def matrix_eigen_decomposition(self, matrix = None):
		self.eigenvectors_sorted = None
		self.eigenvalues_sorted = None

		if(matrix is not None):
			d, v = np.linalg.eigh(matrix)
			idx_sorted = np.argsort(-np.real(d))	# Sort in descending order of eigenvalues
			d_sorted = np.real(d[idx_sorted])
			
			self.d_normalized = d_sorted/np.sum(d_sorted)
			self.eigenvectors_sorted = v[:, idx_sorted]
			self.eigenvalues_sorted = d_sorted

			self.find_sig_eigenvalues()

			return self.eigenvalues_sig, self.eigenvectors_sig, self.d_normalized
		else:
			print('Compute/Supply covariance matrix first!')
			return

	def find_sig_eigenvalues(self):

		N_eigvalues = 10

		for n_eig in range(N_eigvalues):
			
			perc_var = 100*np.sum(self.d_normalized[0:n_eig])
			
			if(perc_var>=95):
				break
				
		print('No:of eigenvalues to explain {} perc of variance: {}'.format(95, n_eig))

		self.n_sig_eigenvalues = n_eig

		# Chop the eigenvectors to only keep the ones with the most contribution to the variance
		self.eigenvectors_sig = self.eigenvectors_sorted[:,0:self.n_sig_eigenvalues]
		self.eigenvalues_sig = self.eigenvalues_sorted[0:self.n_sig_eigenvalues]

	def save_eigenvectors(self):
		""" Save the significant eigenvectors and associated eigenvalues

		"""
		save_file = self.dataName[:-5] + '_eigenvectors.hdf5'


		with h5py.File(os.path.join(self.analysisFolder, save_file), "w") as f:

			dset = f.create_group('eigenvectors')
			dset.create_dataset('eigenvalues', data = self.eigenvalues_sig)
			dset.create_dataset('eigenvectors', data = self.eigenvectors_sig)


	def load_eigenvectors(self):

		load_file = self.dataName[:-5] + '_eigenvectors.hdf5'

		if(os.path.exists(os.path.join(self.analysisFolder, load_file))):

			with h5py.File(os.path.join(self.analysisFolder, load_file), "r") as f:

				dset = f['eigenvectors']

				self.eigenvalues_sig = dset['eigenvalues'][:]
				self.eigenvectors_sig = dset['eigenvectors'][:]

				self.n_sig_eigenvalues = len(self.eigenvalues_sig)

		else:
			print('Eigenvectors file not found!')

	def plot_shape_modes(self, save = False, save_folder = None):
		'''
			Plot the shape modes obtained from PCA

		'''
		plt.figure(figsize = (4,4))
		length_array = np.linspace(0, 1, len(self.eigenvectors_sig[:,0]))
		for ii in range(self.n_sig_eigenvalues):
			
			plt.plot(length_array, self.eigenvectors_sorted[:, ii], label =' Shape mode {}'.format(ii+1), linewidth = 3)
			
		plt.xlabel('Normalized arc length')
		plt.ylabel('Tangent angle (\phi)')
		plt.title('Shaped modes from eigen-decomposition')
		plt.legend()
		if(save):
			if(save_folder is not None):

				file_path = os.path.join(save_folder, self.subFolder)
			else:
				file_path = self.analysisFolder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_' + '_ShapeModes'
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')

		plt.show()

		
	def project_filament_shapes(self):
		""" 
			Projects a give filament shape onto the shape modes computed using PCA.
			For a time series of shapes, returns a time-series of mode amplitudes for each shape mode.

		"""

		n_times, n_points = np.shape(self.tangent_angles_matrix)

		assert(n_times == self.Nt) # Make sure we have the Covariance matrix for the whole simulation

		self.mode_amplitudes = np.zeros((n_times, self.n_sig_eigenvalues))


		matrix_A = self.eigenvectors_sig # n_points x n_eigvalues
		matrix_A_inv = np.linalg.pinv(matrix_A)

		
		for ii in range(n_times):

			rhs = self.variance_matrix[ii, :]

			amplitudes_lst_sq = np.matmul(matrix_A_inv, rhs) # 
	
			for jj in range(self.n_sig_eigenvalues):
				self.mode_amplitudes[ii, jj] = amplitudes_lst_sq[jj]


	def save_mode_amplitudes(self):
		df_dict = {}
		df_dict['Time'] = self.Time

		for ii in range(self.n_sig_eigenvalues):
			
			df_dict['Mode {} amplitude'.format(ii+1)] = self.mode_amplitudes[:,ii]
			
		df_mode_amplitudes = pd.DataFrame(df_dict)
		

		df_mode_amplitudes.to_csv(os.path.join(self.analysisFolder, self.dataName[:-5]+'_ModeAmplitudes.csv'))
		
	def find_constant_phase_indices(self, phase = 0):
		""" Finds the time point indices of the data that correspond to the same activity phase.
			Only use for periodic activity profiles with a fixed activity timescale.
		"""

		phase_value = phase
		# Smallest phase difference = 2*pi*delta_T/T
		delta_phase = 2*np.pi*np.mean(self.Time[1:]-self.Time[:-1])/self.activity_timescale
		abs_val_array = np.abs(self.derived_data['Phase'] - phase_value)
		constant_phase_mask = abs_val_array <= 0.5*delta_phase
		time_points = np.array(range(0, self.Nt))
		constant_phase_indices = time_points[constant_phase_mask]

		return constant_phase_indices


	def compute_base_tip_angle(self):
		""" Compute the unit vector along the base to tip vector of the filament
		"""
		
		self.derived_data['base tip angle'] = np.zeros(self.Nt)
		self.derived_data['base tip vector'] = np.zeros((3, self.Nt))

		unit_vector_lab = [1, 0, 0] # Unit vector fixed to lab reference frame

		for ii in range(self.Nt):

			head_pos_x = self.R[ii, self.Np-1]
			head_pos_y = self.R[ii, 2*self.Np-1]
			head_pos_z = self.R[ii, 3*self.Np-1]

			base_pos_x = self.R[ii, 0]
			base_pos_y = self.R[ii, self.Np]
			base_pos_z = self.R[ii, 2*self.Np]

			vec_x = head_pos_x - base_pos_x
			vec_y = head_pos_y - base_pos_y
			vec_z = head_pos_z - base_pos_z


			vec_mag = (vec_x**2 + vec_y**2 + vec_z**2)**(1/2)

			vec_x = vec_x/vec_mag
			vec_y = vec_y/vec_mag
			vec_z = vec_z/vec_mag

			self.derived_data['base tip vector'][:,ii] = vec_x, vec_y, vec_z

		self.derived_data['base tip angle'] = np.arctan2(self.derived_data['base tip vector'][1,:], self.derived_data['base tip vector'][0,:])


	def compute_tip_angle(self):

		self.derived_data['tip angle'] = np.zeros(self.Nt)
		self.derived_data['tip unit vector'] = np.zeros((self.dim, self.Nt))
		self.derived_data['tip cosine angle'] = np.zeros((self.Nt))

		for ii in range(self.Nt):

			self.r = self.R[ii, :]
			self.get_separation_vectors()
			tangent_angles = self.compute_tangent_angles()

			self.derived_data['tip angle'][ii] = tangent_angles[-1]

			# self.derived_data['Tip unit vector'][:,ii] = self.dr_hat[:,-1]
			# self.derived_data['Tip cosine angle'][ii] = np.dot(self.dr_hat[:,-1], [1, 0 , 0])





	def compute_arc_length(self):

		self.derived_data['Filament arc length'] = np.zeros((self.Nt))

		for ii in range(self.Nt):

			# Particle positions at time ii
			self.r = self.R[ii,:]

			self.get_separation_vectors()

			self.derived_data['Filament arc length'][ii] = np.sum(self.dr)

	def distance_from_array(self, point, array):

		array = np.array(array)

		dx = point[0] - array[:,0]
		dy = point[1] - array[:,1]
		dz = point[2] - array[:,2]

		distance = (dx**2 + dy**2 + dz**2)**(1/2)

		return distance

	def compute_tip_velocity(self):
		'''
		Calculate the speed of the filament tip

		'''

		self.derived_data['tip speed'] = np.zeros(self.Nt-1)
		for ii in range(self.Nt-1):

			delta_t = self.Time[ii+1] - self.Time[ii]
			v_x = (self.derived_data['head pos x'][ii+1]-self.derived_data['head pos x'][ii])/delta_t
			v_y = (self.derived_data['head pos y'][ii+1]-self.derived_data['head pos y'][ii])/delta_t
			v_z = (self.derived_data['head pos z'][ii+1]-self.derived_data['head pos z'][ii])/delta_t

			self.derived_data['tip speed'][ii] = (v_x**2 + v_y**2 + v_z**2)**(1/2)



	def filament_tip_coverage(self, save = False, overwrite = False):
		"""	Calculates the no:of unique areas covered by the tip of the filament (head). This serves as a metric for 
		search coverage. 
		Pseudocode:
		- Initialize an empty list of unique_positions=[]. 
		- Initialize unique_counter=0
		- Initialize hits_counter = [] (keeps track of no:of times a given point has been visited)
		- For each time point
			- If the distance between this position and all previous positions is >= particle diameter
				- Add the current position to the list
				- Increment unique_counter by 1. 
			- Else if the distance to all previous unique locations is < particle_diameter
				- Increment the hit_counter for the point nearest to the current location by 1. 
		"""
		analysis_type = 'SearchCoverage'
		self.unique_counter = 0
		self.unique_positions = []
		self.unique_position_times = []
		self.hits_counter = {}

		self.unique_counter_time = np.zeros(self.Nt)


		analysis_sub_folder = os.path.join(self.analysisFolder, analysis_type)

		timeseries_file = os.path.join(analysis_sub_folder, self.dataName[:-5] + '_unique_counts_timeseries.csv')
		unique_positions_file = os.path.join(analysis_sub_folder, self.dataName[:-5] + '_unique_positions.csv')


		if(overwrite == False and os.path.exists(timeseries_file) and os.path.exists(unique_positions_file)):
			# If the data exists then load it
			print('Loading data from file...')
			df_unique_count = pd.read_csv(timeseries_file)
			self.unique_counter_time = df_unique_count['Unique positions count']

			df_unique_positions = pd.read_csv(unique_positions_file)

			hits_counter_keys, hits_counter_values = df_unique_positions['ID'], df_unique_positions['Hits']

			self.hits_counter = {hits_counter_keys[ii]:hits_counter_values[ii] for ii in range(len(hits_counter_keys))}

			self.unique_positions = np.zeros((len(df_unique_positions), 3))

			self.unique_positions[:,0] = df_unique_positions['Position X']
			self.unique_positions[:,1] = df_unique_positions['Position Y']
			self.unique_positions[:,2] = df_unique_positions['Position Z']

			self.unique_position_times = df_unique_positions['Time']

		else:
			# If the unique positions data doesnt exist, then caclulate it
			print('Calculating unique positions and count...')

			for ii in range(self.Nt):

				# Particle positions (head/filament-tip position) at time ii
				self.r = [self.R[ii, self.Np-1], self.R[ii, 2*self.Np-1], self.R[ii, 3*self.Np-1] ]

				# print(self.r)
				# Get the separation distance to list of previous unique locations
				if(not self.unique_positions):
					# If list is empty
					self.unique_positions.append(self.r)
					self.unique_position_times.append(self.Time[ii])
					self.unique_counter+=1
					self.hits_counter[self.unique_counter-1]=1
				else:
					# If list is not empty
					# Find the Euclidean distance between current point and list of all previous unique points. 
					distance = self.distance_from_array(self.r, self.unique_positions)

					if(not np.any(distance<=2*self.radius)):
						self.unique_positions.append(self.r)
						self.unique_position_times.append(self.Time[ii])
						self.unique_counter+=1
						self.hits_counter[self.unique_counter-1]=1
					else:

						idx = np.argmin(distance)
						self.hits_counter[idx]+=1

				self.unique_counter_time[ii] = self.unique_counter



			self.unique_positions = np.array(self.unique_positions)
			self.unique_position_times = np.array(self.unique_position_times)

			self.derived_data['unique position count'] = self.unique_counter_time

			print('Total unique positions sampled by tip: {}'.format(self.unique_counter_time[-1]))
			# Save the data
			if(save == True):

				hits_counter_keys = np.array(list(self.hits_counter.keys()))
				hits_counter_values = np.array(list(self.hits_counter.values()))

				self.create_analysis_folder()
				analysis_sub_folder = os.path.join(self.analysisFolder, analysis_type)
				if(not os.path.exists(analysis_sub_folder)):
					os.makedirs(analysis_sub_folder)

				df_unique_count = pd.DataFrame({'Time':self.Time, 'Unique positions count':self.unique_counter_time})
				df_unique_positions = pd.DataFrame({'ID': hits_counter_keys, 
					'Time': self.unique_position_times, 'Hits': hits_counter_values,
					'Position X':self.unique_positions[:,0], 'Position Y':self.unique_positions[:,1], 
					'Position Z':self.unique_positions[:,2]})

				df_unique_count.to_csv(timeseries_file)
				df_unique_positions.to_csv(unique_positions_file)
	def total_tip_distance(self):
		''' Calculate the cumulative distance covered by the filament tip


		'''
		total_distance = 0
		
		disp_array_x = self.derived_data['head pos x'][1:] - self.derived_data['head pos x'][0:-1]
		disp_array_y = self.derived_data['head pos y'][1:] - self.derived_data['head pos y'][0:-1]
		disp_array_z = self.derived_data['head pos z'][1:] - self.derived_data['head pos z'][0:-1] 

		total_distance = np.sum((disp_array_x**2 + disp_array_y**2 + disp_array_z**2)**(1/2))

		return total_distance


	def search_efficiency(self):
		""" Search efficiency = Unique sites sampled by tip/Total distance covered by tip
			Run filament_tip_coverage before running this

		"""
		
		total_distance = self.total_tip_distance()
		
		total_unique_locations = self.unique_counter_time[-1]

		return total_unique_locations/total_distance


	def classify_filament_dynamics(self):
		''' Classify the filament dynamics into 1. Periodic or 2. Aperiodic
			If periodic, find the period.

			Returns:

				periodic_flag: bool, with True when the dynamics is periodic, False for Aperiodic
				period: int, Period (multiple of driving or forcing time-scale) over which the dynamics is periodic, None for aperiodic dynamics
		'''
		periodic_flag = False
		min_period = None
		threshold_index = 0

		# Find time points at a constant phase (stroboscopic)
		# In the current activity profile, phase = 0 is start of compression, phase = pi is start of extension
		phase_value = 0
		# Smallest phase difference = 2*pi*delta_T/T
		delta_phase = 2*np.pi*np.mean(self.Time[1:]-self.Time[:-1])/self.activity_timescale
		abs_val_array = np.abs(self.derived_data['Phase'] - phase_value)
		constant_phase_mask = abs_val_array <= 0.5*delta_phase
		time_points = np.array(range(0, self.Nt))
		constant_phase_indices = time_points[constant_phase_mask]

		print(len(constant_phase_indices))
		# Compare filament shapes at two points at constant phase 
		period_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64] # Number of periods over which we want to compare. period = 1 means every cycle.
		below_threshold_array = np.zeros(len(period_array), dtype = 'bool')
		# Threshold value for comparing shapes of filaments. Currently choosen as 10% of the sphere radius. 
		epsilon = 0.1*self.radius

		for period_index, period in enumerate(period_array):

			if(len(constant_phase_indices)-period > 1):
				pair_wise_distance = np.zeros(len(constant_phase_indices)-period)

				for ii in range(len(constant_phase_indices)-period):

					index_a = constant_phase_indices[ii]
					index_b = constant_phase_indices[ii+period]

					filament_a = self.R[index_a, :]
					filament_b = self.R[index_b, :]

					# Calculate the pair-wise distance between the shapes of the two filaments
					distance = self.euclidean_distance(filament_a, filament_b)
					pair_wise_distance[ii] = distance

				# Find if the distance goes below the threshold (and stays there)
				# threshold_index = next((i for i,x in enumerate(pair_wise_distance) if (pair_wise_distance[i]<=epsilon and np.all(pair_wise_distance[i:]<epsilon)) or (pair_wise_distance[i]<=epsilon and np.all(pair_wise_distance[int(len(constant_phase_indices)/2):-1]<epsilon))) , None)
				for i in range(len(pair_wise_distance)-1):
		
					if(pair_wise_distance[i] <=epsilon and pair_wise_distance[i+1]<=epsilon and np.all(pair_wise_distance[-10:]<=epsilon)):
						threshold_index = i
						break
					else:
						threshold_index = None

				if(threshold_index is not None):
					below_threshold_flag = True
					below_threshold_array[period_index] = below_threshold_flag
					periodic_flag = True
				else:
					periodic_flag = False
					threshold_index = 0

			else:
				continue

			if periodic_flag==True:
				# If we have found the min period then we can break from the main loop.
				break

		# Summarize the results

		print(50*'*')
		print('Is the dynamics Periodic? :{}'.format(periodic_flag))
		print(50*'*')

		min_period = next((x for i, x in enumerate(period_array) if below_threshold_array[i]==True), None)

		if(min_period is not None):
			print('The minimum period of the system is {} times the forcing period'.format(min_period))
			print(50*'*')

		return periodic_flag, min_period, threshold_index

	def compute_basetip_angle_at_constant_phase(self, phase_value = 0, skip_cycles =100):

		delta_phase = 2*np.pi*np.mean(self.Time[1:]-self.Time[:-1])/self.activity_timescale
		abs_val_array = np.abs(self.derived_data['Phase'] - phase_value)
		constant_phase_mask = abs_val_array <= 0.5*delta_phase

		time_points = np.array(range(0, self.Nt))
		constant_phase_indices = time_points[constant_phase_mask]

		self.compute_base_tip_angle()

		filament_angles = self.derived_data['base tip angle'][constant_phase_indices[skip_cycles:]]

		# # Get a list of filament tip locations (after skipping half the total number of simulated cycles)
		# filament_locations_x = self.derived_data['head pos x'][constant_phase_indices[int(len(constant_phase_indices)/2):]]
		# filament_locations_y = self.derived_data['head pos y'][constant_phase_indices[int(len(constant_phase_indices)/2):]]

		# # Get the angles that the filament tip reaches at the end of each extension
		# filament_angles =  np.arctan2(filament_locations_y, filament_locations_x)
		
		return filament_angles


	# Energy based metrics:

	# Filament energies (Axial and bending)

	def compute_axial_bending_energy(self):

		self.derived_data['Axial energy'] = np.zeros((self.Nt))
		self.derived_data['Bending energy'] = np.zeros((self.Nt))

		for ii in range(self.Nt):

			self.r = self.R[ii,:]

			self.get_separation_vectors()

			self.derived_data['Axial energy'][ii] = np.sum((self.k/2)*(self.dr - self.b0)**2)

			self.filament.get_bond_angles(self.dr_hat, self.cosAngle)

			self.derived_data['Bending energy'][ii] = np.sum(self.kappa_hat*(1 - self.cosAngle[0:-1]))

	
	
	# Diagnostics/Testing functions
	def compute_self_interaction_forces(self):
		self.derived_data['self-interaction forces'] = np.zeros((self.dim*self.Np, self.Nt))
		
		stride = 100
		for ii, index in enumerate(tqdm(range(0,self.Nt, stride))):

	
			self.r = self.R[index,:]

			self.get_separation_vectors()
			self.F_sc = np.zeros(self.dim*self.Np, dtype = np.double)
			assert(np.sum(self.F_sc[:self.Np])==0)
			assert(np.sum(self.F_sc[self.Np:2*self.Np])==0)
			assert(np.sum(self.F_sc[2*self.Np:3*self.Np])==0)
			self.self_contact_forces(self.r, self.dr, self.dr_hat)

			self.derived_data['self-interaction forces'][:, ii] = self.F_sc

	# def compute_alignment_parameter(self, field_vector = [1,0,0]):

	# 	alignmentParameter = np.zeros((self.Nt))

	# 	field_vector = np.expand_dims(field_vector, axis = 1)

	# 	for ii in range(self.Nt):

	# 		# Set the current filament positions to those from the simulation result at time point ii
	# 		self.r = self.R[ii,:]
	# 		# Get the separation vector based on this position
	# 		self.get_separation_vectors()

	# 		alignmentParameter[ii] = (1/(self.Np-1))*(np.sum((3/2)*self.dotProduct(self.dr_hat, field_vector)**2 - (1/2)))
	# 	plt.figure()
	# 	plt.plot(self.Time, alignmentParameter, 'ro')
	# 	plt.show()

	# 	return alignmentParameter

#-------------------------------------
	# Plotting tools
#-------------------------------------

	def plot_timeseries(self, var, data = None, save = False, save_folder = None, title = '', colors = None):

		x_data = self.Time
		y_data = {}

		for key in var:
			if(data is not None and data[key] and data[key] is not None):
				y_data[key] = data[key]
			else:
				y_data[key] = self.derived_data[key]

		num_plots = len(var)

		if(colors is None):
			cmap = cm.get_cmap('viridis', 255)
			colors = [cmap(ii) for ii in np.linspace(0,1,num_plots)]

		
		if(num_plots>1):
			fig, ax = plt.subplots(nrows = num_plots, ncols=1, sharex = 'row')

			for ii, key in enumerate(var):
				ax[ii].plot(x_data, y_data[key], color = colors[ii], linestyle = '-', label = key)
				ax[ii].set_ylabel(key)
		else:
			plt.figure()

			for ii, key in enumerate(var):
				plt.plot(x_data, y_data[key], color = colors[ii], linestyle = '-', label = key)
				plt.ylabel(key)
		
		plt.legend()
		plt.xlabel('Time')

		if(save):
			if(save_folder is not None):

				file_path = os.path.join(save_folder, self.subFolder)
			else:
				file_path = self.analysisFolder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_' + '_TimeSeries'
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')
			
		
		plt.show()
		

	def plot_scatter(self, var_x, var_y,data_x = None, data_y = None, color_by = 'Time', save_folder = None, save = False):

		title = var_y + ' vs ' + var_x

		if(data_x is None):
			data_x = self.derived_data[var_x]

		if(data_y is None):
			data_y = self.derived_data[var_y]

		if(color_by == 'Time'):
			color = self.Time
		elif color_by in self.derived_data.keys():
			color = self.derived_data[color_by]

		plt.figure()

		ax1 = plt.scatter(data_x, data_y, c = color)
		ax2 = plt.scatter(data_x[0], data_y[0], 20, color ='r')
		plt.xlabel(var_x)
		plt.ylabel(var_y)
		plt.title(title)
		cbar = plt.colorbar(ax1)
		cbar.ax.set_ylabel(color_by)

		if(save):
			if(save_folder is not None):

				file_path = os.path.join(save_folder, self.subFolder)
			else:
				file_path = self.analysisFolder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_' + title + '_ScatterPlot'
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')

		plt.show()

	def plot_phase_portrait(self, var_x, var_y,data_x = None, data_y = None, color_by = 'Time', 
		save_folder = None, save = False, title = [], start_index=0, stop_index = -1):

		title = var_y + ' vs ' + var_x

		if(data_x is None):
			data_x = self.derived_data[var_x]

		if(data_y is None):
			data_y = self.derived_data[var_y]

		if(color_by == 'Time'):
			color = self.Time
		elif color_by in self.derived_data.keys():
			color = self.derived_data[color_by]

		if(stop_index==-1):
			stop_index = self.Nt
		# Plot as a phase portrait
		u = data_x[1:] - data_x[0:-1]
		v = data_y[1:] - data_y[0:-1]

		mask = (u**2 + v**2)**(1/2) > (max(data_x) - min(data_x) - 0.1*(max(data_x) - min(data_x)))

		u[mask], v[mask] = 0,0

		plt.figure(figsize = (8,6))
		if(color_by == 'Time'):
			ax1 = plt.quiver(data_x[start_index:stop_index-1],data_y[start_index:stop_index-1],u[start_index:stop_index],v[start_index:stop_index], color[start_index:stop_index-1], scale_units='xy', angles='xy', scale=1, headwidth = 5)
		else:
			ax1 = plt.quiver(data_x[start_index:stop_index-1],data_y[start_index:stop_index-1],u[start_index:stop_index],v[start_index:stop_index], color[start_index:stop_index-1], scale_units='xy', angles='xy', scale=1, headwidth = 5, cmap = cmocean.cm.phase)


		ax2 = plt.scatter(data_x[0], data_y[0], 50, marker = 'o', color = 'r')

		plt.xlabel(var_x)
		plt.ylabel(var_y)
		plt.title(title)
		cbar = plt.colorbar(ax1)
		cbar.ax.set_ylabel(color_by)

		if(save):
			if(save_folder is not None):

				file_path = os.path.join(save_folder, self.subFolder)
			else:
				file_path = self.analysisFolder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_'+title + '_PhasePortrait'
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			# plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')

		plt.show()


	def plot_tip_position(self):

		
		plt.figure()

		ax1 = plt.scatter(self.R[:, self.Np-1], self.R[:, 2*self.Np-1], c = self.Time)
		ax2 = plt.scatter(self.R[:, 0], self.R[:, self.Np], 20, color ='r')
		plt.xlabel('Filament tip (x)')
		plt.ylabel('Filament tip (y)')
		plt.title('Filament tip trajectory (xy)')
		plt.axis('equal')
		plt.xlim([-1*self.Np*self.b0, 1.5*self.Np*self.b0])
		plt.ylim([-1*self.Np*self.b0, 1*self.Np*self.b0])
		cbar = plt.colorbar(ax1)
		cbar.ax.set_ylabel('Time')
		plt.show()


	def plot_unique_tip_locations(self, color_by = 'count', save = False, save_folder = None):


		hits_total = np.sum(list(self.hits_counter.values()))

		print('Total hits : {}'.format(hits_total))

		position_prob_density = [self.hits_counter[key]/hits_total for key in self.hits_counter.keys()]

		hits_counter_list = [self.hits_counter[key] for key in self.hits_counter.keys()]
		
		plt.figure(figsize = (7,5))
		# ax1 = plt.scatter(unique_positions[:,0], unique_positions[:,1], 40, c = position_prob_density, cmap=cmocean.cm.matter)
		
		if(color_by == 'count'):
			ax1 = plt.scatter(self.unique_positions[:,0], self.unique_positions[:,1], 20, c = hits_counter_list, cmap=cmocean.cm.matter, alpha = 0.75)
		elif (color_by == 'probability'):
			ax1 = plt.scatter(self.unique_positions[:,0], self.unique_positions[:,1], 20, c = position_prob_density, cmap=cmocean.cm.matter, alpha = 0.75)
		elif (color_by == 'first-passage-time'):
			ax1 = plt.scatter(self.unique_positions[:,0], self.unique_positions[:,1], 20, c = self.unique_position_times/self.activity_timescale, cmap=cmocean.cm.matter, alpha = 0.75)




		ax2 = plt.scatter(self.R[:, 0], self.R[:, self.Np], 20, color ='b')

		plt.xlabel('Filament tip (x)')
		plt.ylabel('Filament tip (y)')
		plt.title('Search coverage of filament tip')
		plt.axis('equal')
		plt.xlim([-1*self.Np*self.b0, 1.5*self.Np*self.b0])
		plt.ylim([-1*self.Np*self.b0, 1*self.Np*self.b0])
		cbar = plt.colorbar(ax1)
		cbar.ax.set_ylabel(color_by)
		# cbar.ax.set_ylabel('Probability density')
		if(save):
			if(save_folder is not None):

				file_path = os.path.join(save_folder, self.subFolder)
			else:
				file_path = self.analysisFolder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_' + '_UniqueTipLocations'
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')

		plt.show()

	def plot_tip_scatter_density(self, save = False, save_folder = None, fig_name = None, 
									skip_cycles = 50, plot_unique_locations = False, color_by = 'count'):
		# plt.style.use('dark_background')
		# print('Tip scatter density plot')

		# x = np.array(self.derived_data['head pos x'])
		# y = np.array(self.derived_data['head pos y'])
		# xy = np.vstack([x, y])
		# z = gaussian_kde(xy)(xy)

		# idx = z.argsort()
		# x, y, z = x[idx], y[idx], z[idx]

		# if(fig_name is None):
		# 	fig = plt.figure(figsize = (4,4))
		# else:
		# 	fig = plt.figure(num = fig_name)


		# ax1 = plt.scatter(y, x, c = z[::1], s = 20, edgecolor = None, cmap = SPATIAL_DENSITY_CMAP, rasterized = True)
		# plt.scatter(0, 0, c = 'r', s = 40, edgecolor = None)
		# cbar = plt.colorbar(ax1)
		# cbar.ax.set_ylabel('Density')
		# plt.title('Tip locations colored by local density')



		plt.figure(figsize = (7,5))
		# Overlay filament center lines
		phase_value_array = [0, np.pi]
		
		for phase_value in phase_value_array:
			# phase_value = 0
			# Smallest phase difference = 2*pi*delta_T/T
			delta_phase = 2*np.pi*np.mean(self.Time[1:]-self.Time[:-1])/self.activity_timescale
			abs_val_array = np.abs(self.derived_data['Phase'] - phase_value)
			constant_phase_mask = abs_val_array <= 1*delta_phase
			time_points = np.array(range(0, self.Nt))
			constant_phase_indices = time_points[constant_phase_mask]

			# Remove adjacent time points to prevent double counting of time points at constant phase
			adjacent_mask = (constant_phase_indices[1:]-constant_phase_indices[:-1])==1

			constant_phase_indices = constant_phase_indices[1:][~adjacent_mask]
			
			for ii in constant_phase_indices[skip_cycles:]:
				
				self.r = self.R[ii,:]
				x = self.r[0:self.Np]
				y = self.r[self.Np:2*self.Np]

				# if(ii%stride==0):
				if(phase_value==0):
					plt.plot(y, x, color = COMP_COLOR, linewidth = 1.5, alpha = 0.5, zorder=1)
				elif(phase_value==np.pi):
					plt.plot(y, x, color = EXT_COLOR, linewidth = 1.5, alpha = 0.5, zorder=1)
				else:
					plt.plot(y, x, color = 'k', linewidth = 1.5, alpha = 0.5, zorder=1)


		if(plot_unique_locations):

			hits_total = np.sum(list(self.hits_counter.values()))

			print('Total hits : {}'.format(hits_total))

			position_prob_density = [self.hits_counter[key]/hits_total for key in self.hits_counter.keys()]

			hits_counter_list = [self.hits_counter[key] for key in self.hits_counter.keys()]
			
			
			# ax1 = plt.scatter(unique_positions[:,0], unique_positions[:,1], 40, c = position_prob_density, cmap=cmocean.cm.matter)
			
			if(color_by == 'count'):
				ax1 = plt.scatter(self.unique_positions[:,1], self.unique_positions[:,0], 20, c = hits_counter_list, cmap=SPATIAL_DENSITY_CMAP, alpha = 0.75)
			elif (color_by == 'probability'):
				ax1 = plt.scatter(self.unique_positions[:,1], self.unique_positions[:,0], 20, c = position_prob_density, cmap=SPATIAL_DENSITY_CMAP, alpha = 0.75)
			elif (color_by == 'first-passage-time'):
				ax1 = plt.scatter(self.unique_positions[:,1], self.unique_positions[:,0], 20, c = self.unique_position_times/self.activity_timescale, cmap=SPATIAL_DENSITY_CMAP, alpha = 0.75, zorder=1)

			# Plot filament base
			ax2 = plt.scatter(self.R[:, self.Np], self.R[:, 0], 20, color ='b')

			# plt.xlabel('Filament tip (x)')
			# plt.ylabel('Filament tip (y)')
			# plt.title('Search coverage of filament tip')
			
			plt.clim(0, 500)
			cbar = plt.colorbar(ax1)
			cbar.ax.set_ylabel(color_by)
		# plt.axis('equal')
		plt.axis('equal')
		plt.xlim([-1.25*self.Np*self.b0, 1.25*self.Np*self.b0])
		plt.ylim([-1.25*self.Np*self.b0, 1.25*self.Np*self.b0])
		plt.axis('off')

		if(save == True):
			print('saving figure...')
			if(save_folder is not None):

				file_path = os.path.join(save_folder, self.dataFolder)
			else:
				file_path = self.analysisFolder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_' + '_FilamentShapes_SearchCloud'
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')
		

		plt.show()


	def plot_energy_timeseries(self, save_folder = None):

		self.compute_axial_bending_energy()


		fig, ax1 = plt.subplots(figsize = (10,4))
		color = 'tab:red'
		ax1.plot(self.Time, self.derived_data['Axial energy'], linestyle = '-', linewidth = 1, color = color, alpha = 0.75)
		ax1.set_ylabel('Axial energy', color = color)

		ax2 = ax1.twinx()
		color = 'tab:blue'

		ax2.plot(self.Time, self.derived_data['Bending energy'], linestyle = '-', linewidth = 1, color = color, alpha = 0.75)
		ax2.set_ylabel('Bending energy', color = color)

		ax1.set_xlabel('Time')


		if(save_folder is not None):

			file_path = os.path.join(save_folder, self.subFolder)

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)


			file_name = 'energy_time_series'
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300)
			plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300)

		plt.show()

	def plot_filament_centerlines(self, save_folder = None, save = False, stride = 100, color_by = None):

		title = 'Overlay of filament shapes'

		# calculate the mean filament shape
		self.mean_filament_shape = np.nanmean(self.R, axis = 0)

		stride = stride
		cmap = cm.get_cmap('GnBu', 200)
		if(color_by == 'Cycle'):
			colors = [cmap(ii) for ii in np.linspace(0,1,self.Nt)]
			norm = mpl.colors.Normalize(vmin = int(np.min((self.Time/self.activity_timescale))), vmax = int(np.max(self.Time/self.activity_timescale)))
		elif(color_by == 'Phase'):
			norm = mpl.colors.Normalize(vmin=np.min(self.derived_data['Phase']), vmax=np.max(self.derived_data['Phase']))

		# cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
  #                               norm=norm,
  #                               orientation='horizontal')

		fig, ax1 = plt.subplots(figsize = (5,5))
		for ii in range(self.Nt):			
			self.r = self.R[ii,:]
			x = self.r[0:self.Np]
			y = self.r[self.Np:2*self.Np]

			if(ii%stride==0):
				if(color_by is None):
					cf = ax1.plot(y, x, color = 'w', linewidth = 2.0, alpha = 0.5)
				else:
					cf = ax1.plot(y, x, color = colors[ii], linewidth = 2.0, alpha = 0.5)
		# Plot the mean filament shape
		x_mean = self.mean_filament_shape[0:self.Np]
		y_mean = self.mean_filament_shape[self.Np:2*self.Np]
		ax1.plot(y_mean, x_mean, color = 'r', linewidth=2.0, alpha = 0.75)
		ax1.set_xlabel('X position')
		ax1.set_ylabel('Y position')
		ax1.set_title(title)
		if(color_by is not None):
			fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
				 ax=ax1, orientation='vertical', label=color_by)		# cbar.ax.set_ylabel('Time')
		ax1.set_xlim(-self.L/4, self.L)
		ax1.set_ylim(-self.L/2, self.L/2)
		plt.axis('equal')

		if(save):
			if(save_folder is not None):

				file_path = os.path.join(save_folder, self.subFolder)
			else:
				file_path = self.analysisFolder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_'+title 
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')
	

		plt.show(block = False)

	def plot_tangent_angle_matrix(self, save = False, save_folder = None, start_time = 0, end_time = 1000):
		
		start_index = next((i for i,x in enumerate(self.Time) if x>= start_time), 0)
		end_index = next((i for i,x in enumerate(self.Time) if x>= end_time), len(self.Time))


		title = 'tangent_angles'
		grid_kws = {"width_ratios": (.9, 0.05), "wspace": .1}
		
		fig, (ax, cbar_ax) = plt.subplots(figsize = (10,10), nrows=1, ncols = 2, gridspec_kw = grid_kws)
		
		ax = sns.heatmap(self.tangent_angles_matrix[start_index:end_index, :], ax=ax,
				 cbar_ax=cbar_ax,
				 cbar_kws={"orientation": "vertical"}, cmap = cmocean.cm.thermal)
	
		# Customize the X and Y ticks and labels
		x_ticks = np.array(range(0, 100, 10))
		x_tick_labels = [str(x_tick/100) for x_tick in x_ticks]
		max_ticks = 20
		y_ticks = np.array(range(start_index, end_index, int((end_index - start_index)/max_ticks)))
		y_tick_labels = [str(round(self.Time[ii]/self.activity_timescale, 1)) for ii in y_ticks]

		ax.set_xticks(x_ticks)
		ax.set_xticklabels(x_tick_labels)
		ax.set_yticks(y_ticks)
		ax.set_yticklabels(y_tick_labels)

		ax.set_xlabel('Normalized arc length')
		ax.set_ylabel('Activity cycles')
		ax.set_title('Tangent angles')
		cbar_ax.set_ylabel('Tangent angle')

		if(save):
			if(save_folder is not None):
				file_path = os.path.join(save_folder, self.subFolder)
			else:
				file_path = self.analysisFolder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_'+title+'_'+str(start_time)+'_'+str(end_time) 
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			# plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')

		plt.show()

	def plot_shape_covariance_matrix(self, save = False, save_folder = None):

		title = 'ShapeCovarianceMatrix'
		grid_kws = {"width_ratios": (.9, 0.05), "wspace": .1}
		
		fig, (ax, cbar_ax) = plt.subplots(figsize = (10,10), nrows=1, ncols = 2, gridspec_kw = grid_kws)
		
		ax = sns.heatmap(self.covariance_matrix, ax=ax,
				 cbar_ax=cbar_ax,
				 cbar_kws={"orientation": "vertical"}, cmap = cmocean.cm.haline)
		
		# Customize the X and Y ticks and labels
		x_ticks = np.array(range(0, 100, 10))
		x_tick_labels = [str(x_tick/100) for x_tick in x_ticks]
		y_ticks = np.array(range(0, 100, 10))
		y_tick_labels = [str(y_tick/100) for y_tick in y_ticks]

		ax.set_xticks(x_ticks)
		ax.set_xticklabels(x_tick_labels)
		ax.set_yticks(y_ticks)
		ax.set_yticklabels(y_tick_labels)

		ax.set_xlabel('Arc length (s)')
		ax.set_ylabel('Arc length (s`)')
		ax.set_title(title)

		if(save):
			if(save_folder is not None):
				file_path = os.path.join(save_folder, self.subFolder)
			else:
				file_path = self.analysisFolder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_'+title 
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			# plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')





































