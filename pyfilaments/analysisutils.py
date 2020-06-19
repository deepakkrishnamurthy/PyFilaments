# Utility functions for analyzing filament shapes and dynamics

import numpy as np
from pyfilaments.activeFilaments import activeFilament
import matplotlib.pyplot as plt 


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

		self.findTimePoints()

		print('No:of particles : {}'.format(self.Np))
		print('No:of time points : {}'.format(self.Nt))

	def dotProduct(self, a, b):
		''' 
		Vector dot product of arrays of vectors (eg. nDims x N) where the first axis is the dimensionality of the vector-space
		'''
		# Return shape 1 x N

		c = np.sum(a*b,axis=0)
		print(np.shape(c))
		print(c)
		return  c

	def findTimePoints(self):

		self.Nt, *rest  = np.shape(self.R)
			


	def calculate_axial_strain(self, R = None):

		if(R is not None):


			strain_vector = np.zeros((self.Nt, self.Np - 1))

			
			for ii in range(self.Nt):

				# Set the current filament positions to those from the simulation result at time point ii

				self.r = R[ii,:]

				self.getSeparationVector()

				

				strain_vector[ii,:] = link_distance/self.b0

		# Size (Nt, self.Np - 1)
		return strain_vector



	def BendAngle(self):
		'''
		K = |ˆr01 − ˆrN−1N|/2=sin(θ)
		'''


		bendAngle = np.zeros((self.Nt))

		for ii in range(self.Nt):

			# Set the current filament positions to those from the simulation result at time point ii
			self.r = self.R[ii,:]

			

			# Get the separation vector based on this position
			self.getSeparationVector()


			print(np.shape(self.dr_hat))

			bendAngle[ii] = (0.5)*np.dot(self.dr_hat[:,0] - self.dr_hat[:,-1], self.dr_hat[:,0] - self.dr_hat[:,-1])**(1/2) 


		
		return bendAngle

	def alignmentParameter(self, field_vector = [1,0,0]):

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


	def arclength(self):

		self.arc_length = np.zeros((self.Nt))

		for ii in range(self.Nt):

			# Particle positions at time ii
			self.r = self.R[ii,:]

			self.getSeparationVector()

			self.arc_length[ii] = np.sum(self.dr)


	def euclideanDistance(self, r1, r2):
		'''
			Calculate the Euclidean distance between two filament shapes
			Use this metric to conclude if the simulation has reached steady state.
		'''
		# Reshape the dims*Np x 1 to dims x Np

		r1_matrix = self.reshapeToMatrix(r1)	
		r2_matrix = self.reshapeToMatrix(r2)

		distance = np.sum((r1_matrix - r2_matrix)**2)**(1/2)

		return distance


	# Energy based metrics:

	# def filament_elastic_energy(self):

	# def 























