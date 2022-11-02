# Filament shape generator for pyfilaments
from __future__ import division
import numpy as np
import os
import cmocean
import pickle
import matplotlib.pyplot as plt 
from scipy import signal
from scipy import interpolate
import random
import time
import h5py
from tqdm import tqdm
import imp
from pyfilaments._def import *


def filamentShapeGenerator:
	"""

	Generate filament shape based on given parameters

	--------
	filament: PyFilament object which contains the filament parameters


	"""

	def __init__(self, filament = None):


		self.filament = filament
		self.dim = filament.dim
		self.Np = filament.Np
		self.b0 = filament.b0
		self.L = filament.L

		self.r0 = np.zeros(self.Np*self.dim, dtype = np.double) # Position array that holds the filament shape


	def generate_filament_shape(self, init_condition = None):
		''' 
		Parse the shape parameters supplied in init_conditions and generate the appropriate filament shape
			
		--------------
		init_condition: dictionary that contains the init_condition parameters, these include:
			- shape: Initial filament shape
			- noise_amp: Amplitude of transverse perturbations
			- filament: If initializing from a precomputed filament shape data
		

		'''

		self.init_condition = init_condition # dict containing filament initial conditions

		if 'shape' in self.init_condition.keys():

			self.noise_amp = self.init_condition['noise_amp']
			self.shape = 

			if(self.shape=='line'):
				if('angle' in self.init_condition.keys()):
					self.init_angle = self.init_condition['angle']
				else:
					self.init_angle = 0

				self.linear_shape()


			elif(self.shape == 'parametric'):
				# Parametric equation for the filament shape

				pass

			# elif(self.shape == 'sinusoid'):
			# 	if('plane' in init_condition.keys()):
			# 		self.plane = init_condition['plane']
			# 	else:
			# 		self.plane = 'xy'

			# 	if('amplitude' in init_condition.keys()):
			# 		self.amplitude = init_condition['amplitude']
			# 	else:
			# 		self.amplitude = 1e-4

			# 	if('wavelength' in init_condition.keys()):
			# 		self.wavelength = init_condition['wavelength']
			# 	else:
			# 		self.wavelength = self.L

			# 	self.sinusoid_shape()


		elif 'filament' in init_condition.keys():

			# Check if the supplied filament parameters match with the current filament parameters

			assert(np.shape(self.init_condition['filament'])==(self.Np*self.dim), 'Filament shape supplied does not match current filament parameters')

			self.r0 = self.init_condition['filament']


		return self.r0

	def linear_shape(self):
		'''
		Generates a linear filament at an angle to the equilibrium axis
		Noise specifies the size of transverse perturbations in the initial shape.

		'''
		# Initial particle positions and orientations
		for ii in range(self.Np):
			self.r0[ii] = ii*(self.b0)*np.cos(self.init_angle)
			self.r0[self.Np+ii] = ii*(self.b0)*np.sin(self.init_angle) 
			
		# Add random fluctuations in the y-direction
		# y-axis
		self.r0[self.Np+1:2*self.Np] = self.r0[self.Np+1:2*self.Np]+ np.random.normal(0, self.noise_amp, self.Np-1)


	# def arc_shape(self):

	# 	''' Arc shape filament by the tangent angle difference in the start and end of the arc

	# 	'''

	# 	pass


	# def sinusoid_shape(self):
	# 	'''
	# 	 Generates filament with an initial sinusoidal shape

	# 	'''

	# 	if(self.plane == 'xy'):
	# 			first_index = 0
	# 			second_index = self.Np

	# 	elif(self.plane == 'xz'):
	# 		first_index = 0
	# 		second_index = self.xx

	# 	elif(self.plane == 'yz'):
	# 		first_index = self.Np
	# 		second_index = self.xx

	# 	for ii in range(self.Np):

	# 		# The first particle at origin
	# 		if(ii==0):
	# 			self.r0[ii], self.r0[ii+self.Np], self.r0[ii+self.xx] = 0,0,0 

	# 		else:
	# 			self.r0[ii + first_index] = ii*(self.b0)
	# 			self.r0[ii + second_index] = self.amplitude*np.sin(self.r0[ii]*2*np.pi/(self.wavelength))


	def parametric_curve(self):
		''' generate filament with a parametric curve equation.

			x= f(s)
			y = g(s)
			z = h(s) 

			s_i = b_0*i

		'''

		pass





