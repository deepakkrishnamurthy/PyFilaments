'''
Unit testing of the pyfilaments library
'''

import sys
import unittest
import inspect
from numpy.testing import assert_allclose
import numpy as np
import scipy as sp
from pyfilaments.activeFilaments import activeFilament

Tolerance = 1E-8

class FilamentOperationsTest(unittest.TestCase):

	def separation_vectors_test(self):
		pass

	def bond_angles_test(self):
		pass

	def tangent_vectors_test(self):
		pass

	

	def connection_forces_test(self):
		'''
		test that all internal connection forces sum to zero
		'''
		bc = {0:'free', -1:'free'}
		Np = 64
		fil = activeFilament(dim = 3, Np = Np, radius = 1, b0 = 4, k = 100, S0 = 0, D0 = 0, bc = bc)
		fil.get_separation_vectors()

		fil.filament.connection_forces(fil.dr, fil.dr_hat, fil.F_conn)

		F_sum = np.zeros(3)
		for ii in range(3):
			F_sum[ii] = np.sum(fil.F_conn[ii*Np:(ii+1)*Np])

		self.assertTrue((np.asarray(F_sum) < Tolerance).all(),
                       "Internal connection forces don't sum to zero!")


	def bending_forces_test(self):
		'''
		test that all internal bending forces sum to zero
		'''
		bc = {0:'free', -1:'free'}
		Np = 64
		fil = activeFilament(dim = 3, Np = Np, radius = 1, b0 = 4, k = 100, S0 = 0, D0 = 0, bc = bc)
		fil.get_separation_vectors()
		fil.filament.get_bond_angles(fil.dr_hat, fil.cosAngle)
		fil.filament.bending_forces(fil.dr, fil.dr_hat, fil.cosAngle, fil.F_bending)

		F_sum = np.zeros(3)
		for ii in range(3):
			F_sum[ii] = np.sum(fil.F_bending[ii,:])

		self.assertTrue((np.asarray(F_sum) < Tolerance).all(),
                       "Internal bending forces don't sum to zero!")



