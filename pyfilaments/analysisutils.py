# Utility functions for analyzing filament shapes and dynamics
import os
import numpy as np
from pyfilaments.activeFilaments import activeFilament
import matplotlib.pyplot as plt 
import cmocean

# Figure parameters
from matplotlib import rcParams
from matplotlib import rc
from matplotlib import cm


rc('font', family='sans-serif') 
rc('font', serif='Helvetica') 
rc('text', usetex='false') 
rcParams.update({'font.size': 12})

class analysisTools(activeFilament):

	def __init__(self, filament = None, file = None):

		# Initialize the parent activeFilament class so its attr are available
		activeFilament.__init__(self)

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

	def dotProduct(self, a, b):
		''' 
		Vector dot product of arrays of vectors (eg. nDims x N) where the first axis is the dimensionality of the vector-space
		'''
		# Return shape 1 x N

		c = np.sum(a*b,axis=0)
		return  c

	def find_num_time_points(self):

		self.Nt, *rest  = np.shape(self.R)
			


	def compute_axial_strain(self, R = None):

		if(R is not None):


			strain_vector = np.zeros((self.Nt, self.Np - 1))

			
			for ii in range(self.Nt):

				# Set the current filament positions to those from the simulation result at time point ii

				self.r = R[ii,:]

				self.getSeparationVector()

				

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

			
			self.getSeparationVector()  # Get the separation vector based on this position


			bendAngle[ii] = (0.5)*np.dot(self.dr_hat[:,0] - self.dr_hat[:,-1], self.dr_hat[:,0] - self.dr_hat[:,-1])**(1/2) 


		
		return bendAngle

	def compute_alignment_parameter(self, field_vector = [1,0,0]):

		alignmentParameter = np.zeros((self.Nt))

		field_vector = np.expand_dims(field_vector, axis = 1)

		for ii in range(self.Nt):

			# Set the current filament positions to those from the simulation result at time point ii
			self.r = self.R[ii,:]

			

			# Get the separation vector based on this position
			self.getSeparationVector()

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

			self.getSeparationVector()

			self.derived_data['Filament arc length'][ii] = np.sum(self.dr)

	def distance_from_array(self, point, array):

		array = np.array(array)

		dx = point[0] - array[:,0]
		dy = point[1] - array[:,1]
		dz = point[2] - array[:,2]

		distance = (dx**2 + dy**2 + dz**2)**(1/2)

		return distance


	def filament_tip_coverage(self):
		'''
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
		'''
		self.unique_counter = 0
		self.unique_positions = []
		self.hits_counter = {}

		self.unique_counter_time = np.zeros(self.Nt)

		for ii in range(self.Nt):

			# Particle positions at time ii
			self.r = [self.R[ii, self.Np-1], self.R[ii, 2*self.Np-1], self.R[ii, 3*self.Np-1] ]

			# print(self.r)
			# Get the separation distance to list of previous unique locations
			if(not self.unique_positions):
				# If list is empty
				self.unique_positions.append(self.r)
				self.unique_counter+=1
				self.hits_counter[self.unique_counter-1]=1
			else:
				# If list is not empty
				# Find the Euclidean distance between current point and list of all previous unique points. 
				distance = self.distance_from_array(self.r, self.unique_positions)

				if(not np.any(distance<=2*self.radius)):
					self.unique_positions.append(self.r)
					self.unique_counter+=1
					self.hits_counter[self.unique_counter-1]=1
				else:

					idx = np.argmin(distance)
					self.hits_counter[idx]+=1

			self.unique_counter_time[ii] = self.unique_counter



		self.unique_positions = np.array(self.unique_positions)



	# Energy based metrics:

	# def filament_elastic_energy(self):

	# def 

	def compute_head_orientation(self):

		self.derived_data['Tip unit vector'] = np.zeros((self.dim, self.Nt))

		self.derived_data['Tip cosine angle'] = np.zeros((self.Nt))

		# Find the tangent angle at the filament tip
		for ii in range(self.Nt):

			self.r = self.R[ii,:]

			self.getSeparationVector() 	

			self.derived_data['Tip unit vector'][:,ii] = self.dr_hat[:,-1]
			self.derived_data['Tip cosine angle'][ii] = np.dot(self.dr_hat[:,-1], [1, 0 , 0])

	# Filament energies (Axial and bending)

	def compute_axial_bending_energy(self):

		self.derived_data['Axial energy'] = np.zeros((self.Nt))
		self.derived_data['Bending energy'] = np.zeros((self.Nt))

		for ii in range(self.Nt):

			self.r = self.R[ii,:]

			self.getSeparationVector()

			self.derived_data['Axial energy'][ii] = np.sum((self.k/2)*(self.dr - self.b0)**2)

			self.getBondAngles()

			self.derived_data['Bending energy'][ii] = 10*self.kappa_hat*(1 - self.cosAngle[0]) + np.sum(self.kappa_hat*(1 - self.cosAngle[1:-1]))

#-------------------------------------
	# Plotting tools
#-------------------------------------

	def plot_timeseries(self, var, data = None, save_folder = None, title = '', colors = None):

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

		if(save_folder is not None):

			file_path = os.path.join(save_folder, self.sub_folder)

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = title + '_TimeSeries'
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')
			
		plt.xlabel('Time')
		plt.show()
		

	def plot_scatter(self, var_x, var_y,data_x = None, data_y = None, color_by = 'Time', save_folder = None):

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

		if(save_folder is not None):

			file_path = os.path.join(save_folder, self.sub_folder)

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = title + '_ScatterPlot'
			plt.savefig(os.path.join(file_path, file_name + '.png'), dpi = 300, bbox_inches = 'tight')
			plt.savefig(os.path.join(file_path, file_name + '.svg'), dpi = 300, bbox_inches = 'tight')

		plt.show()

	def plot_phase_portrait(self, var_x, var_y,data_x = None, data_y = None, color_by = 'Time', save_folder = None, title = []):

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

		mask = (u**2 + v**2)**(1/2) > 2*np.pi-0.2

		u[mask], v[mask] = 0,0

		plt.figure(figsize = (8,6))
		ax1 = plt.quiver(data_x[:-1],data_y[:-1],u,v, color[:clip_point], scale_units='xy', angles='xy', scale=1)
		ax2 = plt.scatter(data_x[0], data_y[0], 50, marker = 'o', color = 'r')

		plt.xlabel(var_x)
		plt.ylabel(var_y)
		plt.title(title)
		cbar = plt.colorbar(ax1)
		cbar.ax.set_ylabel(color_by)

		if(save_folder is not None):

			file_path = os.path.join(save_folder, self.sub_folder)

			if(not os.path.exists(file_path)):
				os.makedirs(file_path)

			file_name = title + '_PhasePortrait'
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


	def plot_unique_tip_locations(self):

		hits_total = np.sum(list(self.hits_counter.values()))

		print('Total hits : {}'.format(hits_total))

		position_prob_density = [self.hits_counter[key]/hits_total for key in self.hits_counter.keys()]

		hits_counter_list = [self.hits_counter[key] for key in self.hits_counter.keys()]
		
		plt.figure()
		# ax1 = plt.scatter(unique_positions[:,0], unique_positions[:,1], 40, c = position_prob_density, cmap=cmocean.cm.matter)
		
		ax1 = plt.scatter(self.unique_positions[:,0], self.unique_positions[:,1], 20, c = hits_counter_list, cmap=cmocean.cm.matter)
		ax2 = plt.scatter(self.R[:, 0], self.R[:, self.Np], 20, color ='b')

		plt.xlabel('Filament tip (x)')
		plt.ylabel('Filament tip (y)')
		plt.title('Search coverage of filament tip')
		plt.axis('equal')
		plt.xlim([-1*self.Np*self.b0, 1.5*self.Np*self.b0])
		plt.ylim([-1*self.Np*self.b0, 1*self.Np*self.b0])
		cbar = plt.colorbar(ax1)
		cbar.ax.set_ylabel('Count')
		# cbar.ax.set_ylabel('Probability density')
		plt.show()

	def plot_energy_timeseries(self, save_folder = None):

		self.compute_axial_bending_energy()


		fig, ax1 = plt.subplots(figsize = (10,4))
		color = 'tab:red'
		ax1.plot(self.Time, self.axial_energy, linestyle = '-', linewidth = 1, color = color, alpha = 0.75)
		ax1.set_ylabel('Axial energy', color = color)

		ax2 = ax1.twinx()
		color = 'tab:blue'

		ax2.plot(self.Time, self.bending_energy, linestyle = '-', linewidth = 1, color = color, alpha = 0.75)
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
































