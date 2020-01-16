# Utility functions for analyzing filament shapes and dynamics

import numpy as np
from activeFilaments import activeFilament

class analysisTools(activeFilament):

	def __init__(self, file = None):

		super().__init__(parent)

		#
		if(file is not None):
			self.loadData(file)


	def calculateStrain(self, R = None):

		if(R is not None):

			Nt, *rest  = np.shape(self.R)

			

			strain_vector = np.zeros((Nt, self.Np - 1))

			pos_vector = np.zeros((self.dim, self.Np))
			
			for ii in range(Nt):

				pos_vector[0,:] = self.R[ii, :self.Np]
				pos_vector[1,:] = self.R[ii, self.Np:2*self.Np]
				pos_vector[2,:] = self.R[ii, 2*self.Np:3*self.Np]

				diff = np.diff(pos_vector, axis = 1)

				link_distance = (diff[0,:]**2 + diff[1,:]**2 + diff[2,:]**2)**(1/2)

				strain_vector[ii,:] = link_distance/self.b0

		# Size (Nt, self.Np - 1)
		return strain_vector


	def filamentBend(self, R = None):

		if(R is not None):


