# Utility functions for analyzing filament shapes and dynamics
import os
import numpy as np
from pyfilaments.activeFilaments import activeFilament
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

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

		# If a saved file is supplied, then load it into memory.
		elif(file is not None):
			self.load_data(file)

		# Get the root path for the saved data
		self.rootFolder, self.dataName = os.path.split(file)

		print('Root path: ', self.rootFolder)
		print('Data file', self.dataName)

		self.find_num_time_points()

		print('No:of particles : {}'.format(self.Np))
		print('No:of time points : {}'.format(self.Nt))

		# Find time points corresponding to activity phase =0 and activity phase = pi
		self.derived_data['Phase'] = 2*np.pi*(self.Time%self.activity_timescale)/self.activity_timescale

		self.derived_data['head pos x'] = self.R[:, self.Np-1]
		self.derived_data['head pos y'] = self.R[:, 2*self.Np-1]
		self.derived_data['head pos z'] = self.R[:, 3*self.Np-1]

		# Sub-folder in which to save analysis data and plots
		self.sub_folder = 'Analysis'

		# Create a sub-folder to save Analysis results
		self.analysis_folder = os.path.join(self.rootFolder, self.sub_folder)

		
	
	def create_analysis_folder(self):
		if(not os.path.exists(self.analysis_folder)):
			os.makedirs(self.analysis_folder)


            

	def dotProduct(self, a, b):
		''' 
		Vector dot product of arrays of vectors (eg. nDims x N) where the first axis is the dimensionality of the vector-space
		'''
		# Return shape 1 x N

		c = np.sum(a*b,axis=0)
		return  c

	def find_num_time_points(self):

		self.Nt, *rest  = np.shape(self.R)
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



	def compute_bend_angles(self):
		'''
		K = |ˆr01 − ˆrN−1N|/2=sin(θ)
		'''
		bendAngle = np.zeros((self.Nt))

		for ii in range(self.Nt):			
			self.r = self.R[ii,:]  # Set the current filament positions to those from the simulation result at time point ii			
			self.get_separation_vectors()  # Get the separation vector based on this position
			bendAngle[ii] = (0.5)*np.dot(self.dr_hat[:,0] - self.dr_hat[:,-1], self.dr_hat[:,0] - self.dr_hat[:,-1])**(1/2) 

		return bendAngle

	def compute_alignment_parameter(self, field_vector = [1,0,0]):

		alignmentParameter = np.zeros((self.Nt))

		field_vector = np.expand_dims(field_vector, axis = 1)

		for ii in range(self.Nt):

			# Set the current filament positions to those from the simulation result at time point ii
			self.r = self.R[ii,:]
			# Get the separation vector based on this position
			self.get_separation_vectors()

			alignmentParameter[ii] = (1/(self.Np-1))*(np.sum((3/2)*self.dotProduct(self.dr_hat, field_vector)**2 - (1/2)))
		plt.figure()
		plt.plot(self.Time, alignmentParameter, 'ro')
		plt.show()

		return alignmentParameter


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
		"""
		calculates the no:of unique areas covered by the tip of the filament (head). This serves as a metric for 
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

	def self_contact_forces(self, r, dr, dr_hat):

		Np = self.Np
		xx = 2*Np
		b0 = self.b0
		radius = self.radius
		ljrmin = 3.0*radius
		ljeps = 0.1
		min_distance = (b0*b0) + 4.0*radius*radius

		for i in range(Np-1):
			f_x=0.0; f_y=0.0; f_z=0.0;
			for j in range(Np-1):
				dx = r[i   ] - r[j+1]
				dy = r[i+Np] - r[j+1+Np]
				dz = r[i+xx] - r[j+1+xx] 
				dr_ij1 = dx*dx + dy*dy + dz*dz
			
				dx = r[i+1] - r[j   ]
				dy = r[i+1+Np] - r[j+Np]
				dz = r[i+1+xx] - r[j+xx] 
				dr_i1j = dx*dx + dy*dy + dz*dz
			
				dx = r[i+1] - r[j+1]
				dy = r[i+1+Np] - r[j+1+Np]
				dz = r[i+1+xx] - r[j+1+xx] 
				dr_i1j1 = dx*dx + dy*dy + dz*dz

				dx = r[i   ] - r[j   ]
				dy = r[i+Np] - r[j+Np]
				dz = r[i+xx] - r[j+xx] 
				dr_ij = dx*dx + dy*dy + dz*dz
			
				# if i!=j and abs(i-j)>int(Np/4) and (dr_ij < min_distance or dr_ij1 < min_distance or dr_i1j < min_distance or dr_i1j1 < min_distance):  # No need to check for interactions between very close spheres
				if i!=j and abs(i-j)> 4 and (dr_ij < min_distance or dr_ij1 < min_distance or dr_i1j < min_distance or dr_i1j1 < min_distance):
				# calculate the minimum distance between the two rod-segments that span i, i+1 and j, j+1
					
					h_i, h_j, dri_dot_drj = self.perp_separation(r, dr_hat, dr, i, j)
					
					if(dri_dot_drj == 0):
						f_x = 0
						f_y = 0
						f_z = 0
					else:
						if(h_i >= 0 and h_i <= dr[i] and h_j >= 0 and h_j <= dr[j]):
							gamma_min_final, delta_min_final = 0,0 
						else:
							gamma_min_final, delta_min_final = self.parallel_separation(dri_dot_drj, dr[i], dr[j], h_i, h_j)

						dmin_x = r[i   ] - r[j   ] + (h_i + gamma_min_final)*dr_hat[0, i] - (h_j + delta_min_final)*dr_hat[0, j]
						dmin_y = r[i+Np] - r[j+Np] + (h_i + gamma_min_final)*dr_hat[1, i] - (h_j + delta_min_final)*dr_hat[1, j]
						dmin_z = r[i+xx] - r[j+xx] + (h_i + gamma_min_final)*dr_hat[2, i] - (h_j + delta_min_final)*dr_hat[2, j]

						dmin_2	= dmin_x*dmin_x +dmin_y*dmin_y + dmin_z*dmin_z

						# epsilon = (2*radius - sqrt(dmin_2))

						idr     = 1.0/(dmin_2**(1/2))
						rminbyr = ljrmin*idr 
						fac   = ljeps*(np.power(rminbyr, 12) - np.power(rminbyr, 6))*idr*idr
						# if(epsilon>0):
						f_x += fac*dmin_x
						f_y += fac*dmin_y
						f_z += fac*dmin_z

						# Force on colloid i
					torque_scale_factor = (h_i + gamma_min_final)/dr[i]
					self.F_sc[i] += f_x*(1 - torque_scale_factor)
					self.F_sc[i+Np] += f_y*(1 - torque_scale_factor)
					self.F_sc[i+xx] += f_z*(1 - torque_scale_factor)

					# Force on colloid i+1 
					self.F_sc[i + 1] += f_x*torque_scale_factor
					self.F_sc[i+1+Np] += f_y*torque_scale_factor
					self.F_sc[i+1+xx] += f_z*torque_scale_factor

					# torque_scale_factor = (h_j + delta_min_final)/dr[j]
					# # Force on colloid j
					# F_sc[j] = -f_x*(1-torque_scale_factor)
					# F_sc[j+Np] = -f_y*(1-torque_scale_factor)
					# F_sc[j+xx] = -f_z*(1-torque_scale_factor)

					# # Force on colloid j+1 
					# F_sc[j + 1] = -f_x*torque_scale_factor
					# F_sc[j+1+Np] = -f_y*torque_scale_factor
					# F_sc[j+1+xx] = -f_z*torque_scale_factor

				

	def perp_separation(self, r, dr_hat, dr, i, j):
		""" 
			Takes as input two line-sgments and outputs the minimal 3D separation vector (normal to both lines)
			Returns: h_i, h_j (parameters along the line that determine the minimum separation vector)
		"""
		Np = self.Np 
		xx = 2*Np
		
		
		dri_dot_drj = 0; dri_dot_r = 0; drj_dot_r = 0;
		for index in range(3):
			dri_dot_drj += dr_hat[index, i]*dr_hat[index, j]
			dri_dot_r += dr_hat[index, i]*(r[i + index*Np] - r[j + index*Np])
			drj_dot_r += dr_hat[index, j]*(r[i + index*Np] - r[j + index*Np])
			
		if(abs(dri_dot_drj - 1)<1e-6):
			# Lines are parallel
			return 0, 0, 0
		else:
			h_i = (drj_dot_r*dri_dot_drj - dri_dot_r)/(dr[i]*(1 - dri_dot_drj*dri_dot_drj))
			h_j = (drj_dot_r - dri_dot_drj*dri_dot_r)/(dr[j]*(1 - dri_dot_drj*dri_dot_drj))

			return h_i, h_j, dri_dot_drj
	def parallel_separation(self, dri_dot_drj, l_i, l_j, h_i, h_j):
		"""
		Adapted from Allen et al. Adv. Chem. Phys. Vol LXXXVI, 1003, p.1.

		Takes input of two line-segments and returns the shortest distance and parameters giving the shortest distance vector 
		in the plane parallel to both line segments
		Returns: lambda_min, delta_min
		"""
		gamma_1 = -h_i 
		gamma_2 = -h_i + l_i
		gamma_m = gamma_1
		# Choose the line closest to the origin
		if(gamma_1*gamma_1 > gamma_2*gamma_2):
			gamma_m = gamma_2

		delta_1 = -h_j
		delta_2 = -h_j +l_j
		delta_m = delta_1

		if(delta_1*delta_1 > delta_2*delta_2):
			delta_m = delta_2

		# Optimize delta on gamma_m
		gamma_min = gamma_m
		delta_min = gamma_m*dri_dot_drj

		if(delta_min + h_j >= 0 and delta_min + h_j <= l_j):
			delta_min = delta_min
		else:
			delta_min = delta_1
			a1 = delta_min - delta_1
			a2 = delta_min - delta_2
			if(a1*a1 > a2*a2):
				delta_min = delta_2

		# Distance at this gamma and delta value
		f1 = gamma_min*gamma_min + delta_min*delta_min - 2*gamma_min*delta_min*dri_dot_drj
		gamma_min_final = gamma_min
		delta_min_final = delta_min

		# Now choose the line delta_m and optimize gamma
		delta_min = delta_m
		gamma_min = delta_m*dri_dot_drj

		if(gamma_min + h_i >= 0 and gamma_min + h_i <= l_i):
			gamma_min = gamma_min
		else:
			gamma_min = gamma_1
			b1 = gamma_min - gamma_1
			b2 = gamma_min - gamma_2
			if(b1*b1 > b2*b2):
				gamma_min = gamma_2

		f2 = gamma_min*gamma_min + delta_min*delta_min - 2*gamma_min*delta_min*dri_dot_drj
		
		if(f1 < f2):
			pass
		else:
			delta_min_final = delta_min
			gamma_min_final = gamma_min

		return gamma_min_final, delta_min_final
	
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

	def plot_filament_centerlines(self, save_folder = None, save = False):

		title = 'Overlay of filament shapes'
		stride = 50
		cmap = cm.get_cmap('viridis', 255)
		colors = [cmap(ii) for ii in np.linspace(0,1,self.Nt)]
		norm = mpl.colors.Normalize(vmin=np.min(self.Time), vmax=np.max(self.Time))

		# cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
  #                               norm=norm,
  #                               orientation='horizontal')

		fig, ax1 = plt.subplots(figsize = (8,8))
		for ii in range(self.Nt):			
			self.r = self.R[ii,:]
			if(ii%stride==0):
				cf = ax1.plot(self.r[0:self.Np], self.r[self.Np:2*self.Np], color = colors[ii], linewidth=2.0, alpha = 0.5)

		ax1.set_xlabel('X position')
		ax1.set_ylabel('Y position')
		ax1.set_title(title)
		fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax1, orientation='vertical', label='Time')		# cbar.ax.set_ylabel('Time')
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


































