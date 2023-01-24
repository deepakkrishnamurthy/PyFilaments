from __future__ import division
import numpy as np
import os
import cmocean
import pickle
import matplotlib.pyplot as plt 
from scipy import signal
from scipy import interpolate
from scipy.stats import lognorm
import random
import time
import h5py
from tqdm import tqdm
import imp
from pyfilaments._def import *


class activityPatternGenerator:

	def __init__(self, activity = None):

		self.activity = activity

		# Parse the activity parameters
		if(self.activity is not None):
			self.activity_type = self.activity['type']

			if(self.activity_type == 'square-wave'):
				self.activity_timescale = self.activity['activity time scale']
				self.duty_cycle = self.activity['duty_cycle']
				self.start_phase = self.activity['start phase']

			elif(self.activity_type == 'poisson'):
				# Define variables related to simulating Poisson process
				self.activity_timescale = self.activity['activity time scale']
				self.duty_cycle = self.activity['duty_cycle']
				self.T_ext_mean = self.duty_cycle*self.activity_timescale
				self.T_comp_mean = (1 - self.duty_cycle)*self.activity_timescale
				self.lambda_ext = 1/self.T_ext_mean
				self.lambda_comp = 1/self.T_comp_mean
				self.compression_in_progress = True
				self.extension_in_progress = False
				self.T_ext_start = 0
				self.T_comp_start = 0
				self.current_cycle = 0
				self.t_previous = 0

				# Reset state
				self.t_previous = 0
				self.compression_in_progress = True
				self.extension_in_progress = False
				self.T_ext_start = 0
				self.T_comp_start = 0
				self.current_cycle = 0

			elif(self.activity_type == 'normal'):
				self.activity_timescale = self.activity['activity time scale']
				self.noise_scale = self.activity['noise_scale']
				self.duty_cycle = self.activity['duty_cycle']
				self.n_cycles = self.activity['n_cycles']
				self.T_ext_mean = self.duty_cycle*self.activity_timescale
				self.T_comp_mean = (1 - self.duty_cycle)*self.activity_timescale
				self.curr_phase = 'comp'
				
				self.t_start = 0
				self.counter = 0

				# We create random times for one extra cycle so as to handle cases where Tf is higher than the mean value of n_cycles*(T_ext_mean + T_comp_mean)
				self.T_ext = np.random.normal(loc = self.T_ext_mean, scale = self.noise_scale*self.T_ext_mean, size = self.n_cycles) 
				self.T_comp = np.random.normal(loc = self.T_comp_mean , scale = self.noise_scale*self.T_comp_mean, size = self.n_cycles)
				# Reset Tf based on the actual T-ext and T_comp values
				self.Tf = np.sum(self.T_ext + self.T_comp)
				

			elif self.activity_type == 'biphasic':

				self.initialized = False
				# Biphasic activity where the activity-timescale 
				# or the activity strength can switch between two values
				self.start_phase = self.activity['start phase']
				self.activity_states = ['slow', 'fast']

				self.counter = {'slow':0, 'fast':0}

				self.N_cycles = {'slow':self.activity['N_cycles']['slow'], 'fast':self.activity['N_cycles']['fast']}

				self.activity_timescale_biphasic = {'slow':self.activity['activity time scale']['slow'], 'fast':self.activity['activity time scale']['fast']}

				self.curr_state = self.activity['start_state'] # Can be 1: "fast" or 0:"slow"

				self.duty_cycle = self.activity['duty_cycle']

				self.activity_timescale = self.activity_timescale_biphasic[self.curr_state]

				self.curr_activity = -1
				self.prev_activity = -1

				self.t_start = 0

				self.initialized = True

			elif self.activity_type == 'lognormal':

				self.activity_timescale = self.activity['activity time scale']

				# Standard deviations of the lognormal distributions
				self.sigma_ext = self.activity['sigma extension']
				self.sigma_comp = self.activity['sigma compression']


				self.n_cycles = self.activity['n_cycles']

				self.curr_phase = 'comp'
				
				self.t_start = 0
				self.counter = 0

				self.T_ext_median = self.activity_timescale
				self.T_comp_median = self.activity_timescale/EXT_COMP_SCALEFACTOR

				rng = np.random.default_rng()

				# Draw from the distribution to create a series of extension and compression durations
				self.T_ext = lognorm.rvs(self.sigma_ext, 0, self.T_ext_median, size = self.n_cycles)
				self.T_comp = lognorm.rvs(self.sigma_comp, 0, self.T_comp_median, size = self.n_cycles)

				# self.T_ext = rng.lognormal(mean = np.log(self.T_ext_median), sigma = self.sigma_ext, size = self.n_cycles) 
				# self.T_comp = rng.lognormal(mean = np.log(self.T_comp_median) , sigma = self.sigma_comp, size = self.n_cycles)
				# Reset Tf based on the actual T-ext and T_comp values
				self.Tf = np.sum(self.T_ext + self.T_comp)


	def square_wave_activity(self, t):
		''' Output a square-wave profile based on a cycle time-scale and duty-cycle
		'''
		phase = (t + self.activity_timescale*self.start_phase/(2*np.pi))%self.activity_timescale 
		if(phase > self.activity_timescale*self.duty_cycle):
			return 1
		elif phase < self.activity_timescale*self.duty_cycle:
			return -1
		else:
			return 0

	def poisson_activity(self, t):
		''' Output activity pattern as a Poisson process
		'''
		delta_t = t - self.t_previous
		self.t_previous = t
		if self.compression_in_progress:
			rand_num = np.random.uniform()
			
			if(rand_num < self.lambda_comp*delta_t):
				self.extension_in_progress = True
				self.compression_in_progress = False
				self.T_ext_start = t
				# self.T_comp_poisson[current_cycle] = t - self.T_comp_start
				return 1
			else:
				return -1
			
		elif self.extension_in_progress:
		
			rand_num = np.random.uniform()
			if(rand_num < self.lambda_ext*delta_t):
				self.extension_in_progress = False
				self.compression_in_progress = True
				self.T_comp_start = t
				# self.T_ext_poisson[current_cycle] = t - self.T_ext_start
				self.current_cycle += 1
				return -1
			else:
				return 1

	def normal_activity(self, t):

		t_elapsed = t - self.t_start

		if self.curr_phase == 'comp':
			if t_elapsed >=self.T_comp[self.counter]:
				self.curr_phase = 'ext'
				self.t_start = np.copy(t)
		elif self.curr_phase == 'ext':
			if t_elapsed >= self.T_ext[self.counter]:
				self.curr_phase = 'comp'
				self.t_start = np.copy(t)
				self.counter+=1

		if self.curr_phase == 'comp':
			return -1
		elif self.curr_phase == 'ext':
			return 1


	def lognormal_activity(self, t):

		t_elapsed = t - self.t_start

		if self.curr_phase == 'comp':
			if t_elapsed >=self.T_comp[self.counter]:
				self.curr_phase = 'ext'
				self.t_start = np.copy(t)
		elif self.curr_phase == 'ext':
			if t_elapsed >= self.T_ext[self.counter]:
				self.curr_phase = 'comp'
				self.t_start = np.copy(t)
				self.counter+=1

		if self.curr_phase == 'comp':
			return -1
		elif self.curr_phase == 'ext':
			return 1


	def reset_normal_activity(self):
		# Reset state
		self.curr_phase = 'comp'
		self.t_start = 0
		self.counter = 0

	def reset_lognormal_activity(self):
		# Reset state
		self.curr_phase = 'comp'
		self.t_start = 0
		self.counter = 0


	def reset_biphasic_activity(self):
		self.curr_state = self.activity['start_state'] # Can be 1: "fast" or 0:"slow"
		self.t_start = 0


	def toggle_start_phase(self):

		if self.start_phase == 0:
			self.start_phase = np.pi
		elif self.start_phase == np.pi:
			self.start_phase = 0


	def biphasic_activity(self, t):	  
			
		 # Set the activity parameter based on current state
		self.activity_timescale = self.activity_timescale_biphasic[self.curr_state]
		
		# Get the square-wave activity profile based on current state
		t_elapsed = t - self.t_start # Elapsed time since the start of current activity phase
		
		self.curr_activity = self.square_wave_activity(t_elapsed)
			
		# print('{}, {}, {}'.format(t, self.curr_state, self.t_start))  

		if self.curr_state == 'slow' and t_elapsed>=self.N_cycles['slow']*self.activity_timescale_biphasic['slow']:
			self.curr_state = 'fast'
			# self.counter['fast'] = 0
			self.t_start = np.copy(t)
			# self.toggle_start_phase()
		elif self.curr_state =='fast' and t_elapsed>=self.N_cycles['fast']*self.activity_timescale_biphasic['fast']:
			self.curr_state = 'slow'
			# self.counter['slow'] = 0
			self.t_start = np.copy(t)
			# self.toggle_start_phase()

		return self.curr_activity

	def biphasic_state(self, t):
		 # Set the activity parameter based on current state
		self.activity_timescale = self.activity_timescale_biphasic[self.curr_state]
		
		# Get the square-wave activity profile based on current state
		t_elapsed = t - self.t_start # Elapsed time since the start of current activity phase
		
		self.curr_activity = self.square_wave_activity(t_elapsed)
			
		# print('{}, {}, {}'.format(t, self.curr_state, self.t_start))  

		if self.curr_state == 'slow' and t_elapsed>=self.N_cycles['slow']*self.activity_timescale_biphasic['slow']:
			self.curr_state = 'fast'
			# self.counter['fast'] = 0
			self.t_start = np.copy(t)
			# self.toggle_start_phase()
		elif self.curr_state =='fast' and t_elapsed>=self.N_cycles['fast']*self.activity_timescale_biphasic['fast']:
			self.curr_state = 'slow'
			# self.counter['slow'] = 0
			self.t_start = np.copy(t)
			# self.toggle_start_phase()

		return self.curr_state



	def activity_function(self):
		""" Sets the activity function based on activity-profile in the simulation.

			This function is called by the activeFilament object
		"""
		print(self.activity_type)

		if(self.activity_type == 'square-wave'):
			return lambda t: self.square_wave_activity(t)
		elif(self.activity_type == 'poisson'):
			return lambda t: self.poisson_activity(t)
		elif(self.activity_type == 'normal'):
			return lambda t: self.normal_activity(t)
		elif (self.activity_type == 'lognormal'):
			return lambda t: self.lognormal_activity(t)
		elif(self.activity_type == 'biphasic'):
			return lambda t: self.biphasic_activity(t)

	def activity_profile(self, time_array):
		"""Return activity profile for a given time array
		"""

		activity_array = np.zeros_like(time_array)

		self.filament_activity = self.activity_function()

		for ii, t in enumerate(time_array):

			activity_array[ii] = self.filament_activity(t)

		return activity_array


	def activity_state_profile(self, time_array):
		""" Return the activity state: "slow" vs "fast" when implementing biphasic filament behaviors
		"""
		activity_state_array = np.empty_like(time_array, dtype=bool)

		for ii, t in enumerate(time_array):

			state = self.biphasic_state(t)

			if state =='slow':
				activity_state_array[ii] = 0
			elif state == 'fast':
				activity_state_array[ii] = 1 

		return activity_state_array



		

