# Utility functions for analyzing filament shapes and dynamics
import os
import numpy as np
from pyfilaments.activeFilaments import activeFilament
import filament.filament as filament_operations
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy import interpolate
import seaborn as sns

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

		print(self.Np)

		# Dict to store derived datasets
		self.derived_data = {'Filament arc length':[], 'Tip unit vector': [], 'Tip cosine angle':[],'Activity phase':[],'Axial energy':[],'Bending energy':[]}

		# Set the attributes to those of the filament on which we are doing analysis. 
		if(filament is not None):
			fil_attr = filament.__dict__
			for key in fil_attr.keys():
				setattr(self, key, getattr(filament, key))

			print(self.kappa_hat_array)
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

			print('No:of particles : {}'.format(self.Np))
			print('No:of time points : {}'.format(self.Nt))

			# Find time points corresponding to activity phase =0 and activity phase = pi
			self.derived_data['Phase'] = 2*np.pi*(self.Time%self.activity_timescale)/self.activity_timescale

			self.derived_data['head pos x'] = self.R[:, self.Np-1]
			self.derived_data['head pos y'] = self.R[:, 2*self.Np-1]
			self.derived_data['head pos z'] = self.R[:, 3*self.Np-1]

			# Get the root path for the saved data
			self.rootFolder, self.dataName = os.path.split(file)

			print('Root path: ', self.rootFolder)
			print('Data file', self.dataName)
			# Sub-folder in which to save analysis data and plots
			self.sub_folder = 'Analysis'

			# Create a sub-folder to save Analysis results
			self.analysis_folder = os.path.join(self.rootFolder, self.sub_folder)

	def create_analysis_folder(self):
		"""	Create a sub-folder to store analysis data and plots
		"""
		if(not os.path.exists(self.analysis_folder)):
			os.makedirs(self.analysis_folder) 
		
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
		print(50*'*')

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
		print('Time step: {}'.format(np.max(self.Time)/self.Nt))
			
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
		
		n_time = int(self.Nt/5)
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
		variance_matrix = self.tangent_angles_matrix - np.tile(self.phi_0, (n_times, 1))
		print(np.shape(variance_matrix))
		assert(np.shape(variance_matrix) == (n_times, n_points))
		self.covariance_matrix = np.matmul(variance_matrix.T, variance_matrix)




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

		self.derived_data['tip speed'] = np.zeros(self.Nt-1)
		for ii in range(self.Nt-1):

			delta_t = self.Time[ii+1] - self.Time[ii]
			v_x = (self.derived_data['head pos x'][ii+1]-self.derived_data['head pos x'][ii])/delta_t
			v_y = (self.derived_data['head pos y'][ii+1]-self.derived_data['head pos y'][ii])/delta_t
			v_z = (self.derived_data['head pos z'][ii+1]-self.derived_data['head pos z'][ii])/delta_t

			self.derived_data['tip speed'][ii] = (v_x**2 + v_y**2 + v_z**2)**(1/2)



	def filament_tip_coverage(self, save = False):
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
			analysis_sub_folder = os.path.join(self.analysis_folder, analysis_type)
			if(not os.path.exists(analysis_sub_folder)):
				os.makedirs(analysis_sub_folder)

			df_unique_count = pd.DataFrame({'Time':self.Time, 'Unique positions count':self.unique_counter_time})
			df_unique_positions = pd.DataFrame({'ID': hits_counter_keys, 
				'Time': self.unique_position_times, 'Hits': hits_counter_values,
				'Position X':self.unique_positions[:,0], 'Position Y':self.unique_positions[:,1], 
				'Position Z':self.unique_positions[:,2]})

			df_unique_count.to_csv(os.path.join(analysis_sub_folder, self.dataName[:-5] + '_unique_counts_timeseries.csv'))
			df_unique_positions.to_csv(os.path.join(analysis_sub_folder, self.dataName[:-5] + '_unique_positions.csv'))

	# Energy based metrics:

	# def filament_elastic_energy(self):

	# def 

	def compute_head_orientation(self):

		self.derived_data['Tip unit vector'] = np.zeros((self.dim, self.Nt))

		self.derived_data['Tip cosine angle'] = np.zeros((self.Nt))

		# Find the tangent angle at the filament tip
		for ii in range(self.Nt):

			self.r = self.R[ii,:]

			self.get_separation_vectors() 	

			self.derived_data['Tip unit vector'][:,ii] = self.dr_hat[:,-1]
			self.derived_data['Tip cosine angle'][ii] = np.dot(self.dr_hat[:,-1], [1, 0 , 0])

	# Filament energies (Axial and bending)

	def compute_axial_bending_energy(self):

		self.derived_data['Axial energy'] = np.zeros((self.Nt))
		self.derived_data['Bending energy'] = np.zeros((self.Nt))

		for ii in range(self.Nt):

			self.r = self.R[ii,:]

			self.get_separation_vectors()

			self.derived_data['Axial energy'][ii] = np.sum((self.k/2)*(self.dr - self.b0)**2)

			self.filament.get_bond_angles(self.dr_hat, self.cosAngle)

			self.derived_data['Bending energy'][ii] = 10*self.kappa_hat*(1 - self.cosAngle[0]) + np.sum(self.kappa_hat*(1 - self.cosAngle[1:-1]))

	
	
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

		print(num_plots)

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

				file_path = os.path.join(save_folder, self.sub_folder)
			else:
				file_path = self.analysis_folder

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

				file_path = os.path.join(save_folder, self.sub_folder)
			else:
				file_path = self.analysis_folder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_' + title + '_ScatterPlot'
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')

		plt.show()

	def plot_phase_portrait(self, var_x, var_y,data_x = None, data_y = None, color_by = 'Time', 
		save_folder = None, save = False, title = []):

		title = var_y + ' vs ' + var_x

		if(data_x is None):
			data_x = self.derived_data[var_x]

		if(data_y is None):
			data_y = self.derived_data[var_y]

		if(color_by == 'Time'):
			color = self.Time
		elif color_by in self.derived_data.keys():
			color = self.derived_data[color_by]

		clip_point = int(self.Nt)
		# Plot as a phase portrait
		u = data_x[1:] - data_x[0:-1]
		v = data_y[1:] - data_y[0:-1]

		mask = (u**2 + v**2)**(1/2) > (max(data_x) - min(data_x) - 1)

		u[mask], v[mask] = 0,0

		plt.figure(figsize = (8,6))
		ax1 = plt.quiver(data_x[:-1],data_y[:-1],u,v, color[:clip_point-1], scale_units='xy', angles='xy', scale=1, headwidth = 5)
		ax2 = plt.scatter(data_x[0], data_y[0], 50, marker = 'o', color = 'r')

		plt.xlabel(var_x)
		plt.ylabel(var_y)
		plt.title(title)
		cbar = plt.colorbar(ax1)
		cbar.ax.set_ylabel(color_by)

		if(save):
			if(save_folder is not None):

				file_path = os.path.join(save_folder, self.sub_folder)
			else:
				file_path = self.analysis_folder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_'+title + '_PhasePortrait'
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')

		plt.show()


	def plot_tip_position(self):

		
		print(len(self.R[:, self.Np]))
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
		
		plt.figure()
		# ax1 = plt.scatter(unique_positions[:,0], unique_positions[:,1], 40, c = position_prob_density, cmap=cmocean.cm.matter)
		
		if(color_by == 'count'):
			ax1 = plt.scatter(self.unique_positions[:,0], self.unique_positions[:,1], 20, c = hits_counter_list, cmap=cmocean.cm.matter)
		elif (color_by == 'probability'):
			ax1 = plt.scatter(self.unique_positions[:,0], self.unique_positions[:,1], 20, c = position_prob_density, cmap=cmocean.cm.matter)

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

				file_path = os.path.join(save_folder, self.sub_folder)
			else:
				file_path = self.analysis_folder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_' + '_UniqueTipLocations'
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

			file_path = os.path.join(save_folder, self.sub_folder)

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
		if(color_by == 'Time'):
			colors = [cmap(ii) for ii in np.linspace(0,1,self.Nt)]
			norm = mpl.colors.Normalize(vmin=np.min(self.Time), vmax=np.max(self.Time))
		elif(color_by == 'Phase'):
			norm = mpl.colors.Normalize(vmin=np.min(self.derived_data['Phase']), vmax=np.max(self.derived_data['Phase']))

		# cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
  #                               norm=norm,
  #                               orientation='horizontal')

		fig, ax1 = plt.subplots(figsize = (8,8))
		for ii in range(self.Nt):			
			self.r = self.R[ii,:]
			if(ii%stride==0):
				if(color_by is None):
					cf = ax1.plot(self.r[0:self.Np], self.r[self.Np:2*self.Np], color = 'w', linewidth=2.0, alpha = 0.5)
				else:
					cf = ax1.plot(self.r[0:self.Np], self.r[self.Np:2*self.Np], color = colors[ii], linewidth=2.0, alpha = 0.5)
		# Plot the mean filament shape
		ax1.plot(self.mean_filament_shape[0:self.Np], self.mean_filament_shape[self.Np:2*self.Np], color = 'r', linewidth=2.0, alpha = 0.75)
		ax1.set_xlabel('X position')
		ax1.set_ylabel('Y position')
		ax1.set_title(title)
		if(color_by is not None):
			fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            	 ax=ax1, orientation='vertical', label='Time')		# cbar.ax.set_ylabel('Time')
		ax1.set_xlim(-self.L/4, self.L)
		ax1.set_ylim(-self.L/2, self.L/2)
		plt.axis('equal')

		if(save):
			if(save_folder is not None):

				file_path = os.path.join(save_folder, self.sub_folder)
			else:
				file_path = self.analysis_folder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_'+title 
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')
	

		plt.show()

	def plot_tangent_angle_matrix(self, save = False, save_folder = None):

		title = 'tangent_angles'
		grid_kws = {"width_ratios": (.9, 0.05), "wspace": .1}
		
		fig, (ax, cbar_ax) = plt.subplots(figsize = (10,10), nrows=1, ncols = 2, gridspec_kw = grid_kws)
		
		ax = sns.heatmap(self.tangent_angles_matrix, ax=ax,
                 cbar_ax=cbar_ax,
                 cbar_kws={"orientation": "vertical"}, cmap = cmocean.cm.thermal)
		
		# Customize the X and Y ticks and labels
		x_ticks = np.array(range(0, 100, 10))
		x_tick_labels = [str(x_tick/100) for x_tick in x_ticks]
		y_ticks = np.array(range(0, int(self.Nt/5), 10*int(self.activity_timescale/self.avg_time_step)))
		y_tick_labels = [str(round(self.Time[ii]/self.activity_timescale)) for ii in y_ticks]

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
				file_path = os.path.join(save_folder, self.sub_folder)
			else:
				file_path = self.analysis_folder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_'+title 
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
				file_path = os.path.join(save_folder, self.sub_folder)
			else:
				file_path = self.analysis_folder

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = self.dataName[:-5] +'_'+title 
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			# plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')





































