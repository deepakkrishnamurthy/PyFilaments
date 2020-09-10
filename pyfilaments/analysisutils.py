# Utility functions for analyzing filament shapes and dynamics

import numpy as np
from pyfilaments.activeFilaments import activeFilament
import matplotlib.pyplot as plt 
import cmocean

# Figure parameters
from matplotlib import rcParams
from matplotlib import rc

rc('font', family='sans-serif') 
rc('font', serif='Helvetica') 
rc('text', usetex='false') 
rcParams.update({'font.size': 12})

class analysisTools(activeFilament):

	def __init__(self, filament = None, file = None):

		# Initialize the parent activeFilament class so its attr are available
		activeFilament.__init__(self)

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

		self.arc_length = np.zeros((self.Nt))

		for ii in range(self.Nt):

			# Particle positions at time ii
			self.r = self.R[ii,:]

			self.getSeparationVector()

			self.arc_length[ii] = np.sum(self.dr)

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

#-------------------------------------
	# Plotting tools
#-------------------------------------

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

	def plot_arclength_timeseries(self):
		plt.figure()
		plt.plot(self.Time, self.arc_length, color = 'k', linestyle = '-')
		plt.xlabel('Time')
		plt.ylabel('Arc length')
		plt.title('Arc length time series')
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


	def plot_coverage_vs_time(self):


		plt.figure()
		# ax1 = plt.scatter(unique_positions[:,0], unique_positions[:,1], 40, c = position_prob_density, cmap=cmocean.cm.matter)
		
		ax1 = plt.plot(self.Time, self.unique_counter_time, color = 'g', linewidth = 2)
		plt.xlabel('Time')
		plt.ylabel('Unique head locations')
		plt.title('Search coverage dynamics')

		plt.show()



























