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
			


	def calculateStrain(self, R = None):

		if(R is not None):


			strain_vector = np.zeros((self.Nt, self.Np - 1))

			pos_vector = np.zeros((self.dim, self.Np))
			
			for ii in range(self.Nt):

				pos_vector[0,:] = self.R[ii, :self.Np]
				pos_vector[1,:] = self.R[ii, self.Np:2*self.Np]
				pos_vector[2,:] = self.R[ii, 2*self.Np:3*self.Np]

				diff = np.diff(pos_vector, axis = 1)

				link_distance = (diff[0,:]**2 + diff[1,:]**2 + diff[2,:]**2)**(1/2)

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












