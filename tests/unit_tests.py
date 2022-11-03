'''
Unit testing for pyfilaments
'''

import sys
import unittest
import inspect
from numpy.testing import assert_allclose
import numpy as np
import scipy as sp
from pyfilaments.activeFilaments import activeFilament

from pyfilaments._def import *


Tol = 1E-8

class FilamentOperationsTest(unittest.TestCase):

	def test_separation_vectors(self):
		pass

	def test_bond_angles(self):
		pass

	def test_tangent_vectors(self):
		pass

	
	def test_connection_forces(self):
		'''
		test that all internal connection forces sum to zero
		'''
		bc = {0:'free', -1:'free'}
		Np = 64
		fil = activeFilament(dim = DIMS, Np = Np, radius = RADIUS, b0 = B0, k = K, S0 = 0, D0 = 0, bc = bc)
		fil.get_separation_vectors()

		fil.filament.connection_forces(fil.dr, fil.dr_hat, fil.F_conn)

		F_sum = np.zeros(DIMS)
		for ii in range(DIMS):
			F_sum[ii] = np.sum(fil.F_conn[ii*Np:(ii+1)*Np])

		print(F_sum)

		self.assertTrue((np.asarray(F_sum) < Tolerance).all(),
                       "Internal connection forces don't sum to zero!")


	def test_bending_forces(self):
		'''
		test that all internal bending forces sum to zero
		'''
		bc = {0:'free', -1:'free'}
		Np = 64
		fil = activeFilament(dim = DIMS, Np = Np, radius = RADIUS, b0 = B0, k = K, S0 = 0, D0 = 0, bc = bc)
		fil.get_separation_vectors()
		fil.filament.get_bond_angles(fil.dr_hat, fil.cosAngle)
		fil.filament.bending_forces(fil.dr, fil.dr_hat, fil.cosAngle, fil.F_bending)

		F_sum = np.zeros(DIMS)
		for ii in range(DIMS):
			F_sum[ii] = np.sum(fil.F_bending[ii,:])

		print(F_sum)
		self.assertTrue((np.asarray(F_sum) < Tolerance).all(),
                       "Internal bending forces don't sum to zero!")



if __name__ == '__main__':

    unittest.main()
