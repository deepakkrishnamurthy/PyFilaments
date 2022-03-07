from __future__ import division
import numpy as np
import os
import cmocean
import pickle
import matplotlib.pyplot as plt 
from scipy import signal
from scipy import interpolate
import random
from datetime import datetime
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
				self.n_cycles = n_cycles
				self.T_ext_mean = self.duty_cycle*self.activity_timescale
				self.T_comp_mean = (1 - self.duty_cycle)*self.activity_timescale
				self.compression_in_progress = True
				self.extension_in_progress = False
				self.T_ext_start = 0
				self.T_comp_start = 0
				self.current_cycle = 0

				self.T_ext = np.random.normal(loc = self.T_ext_mean, scale = self.noise_scale*self.T_ext_mean, size = self.n_cycles)
				self.T_comp = np.random.normal(loc = self.T_comp_mean , scale = self.noise_scale*self.T_comp_mean, size = self.n_cycles)
				# Reset Tf based on the actual T-ext and T_comp values
				Tf = np.sum(self.T_ext + self.T_comp)
				# Reset state
				self.compression_in_progress = True
				self.extension_in_progress = False
				self.T_ext_start = 0
				self.T_comp_start = 0
				self.current_cycle = 0

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

		if self.compression_in_progress:
			t_elapsed = t - self.T_comp_start

			if t_elapsed > self.T_comp[self.current_cycle]:
				self.extension_in_progress = True
				self.compression_in_progress = False
				self.T_ext_start = t
				return 1
			else:
				return -1
			
		elif self.extension_in_progress:
			t_elapsed = t - self.T_ext_start

			if t_elapsed > self.T_ext[self.current_cycle]:
				self.extension_in_progress = False
				self.compression_in_progress = True
				self.T_comp_start = t
				self.current_cycle += 1
				return -1
			else:
				return 1

	def reset_biphasic_activity(self):
		self.counter = {'slow':0, 'fast':0}

		self.curr_state = self.activity['start_state'] # Can be 1: "fast" or 0:"slow"

		self.duty_cycle = self.activity['duty_cycle']

		self.activity_timescale = self.activity_timescale_biphasic[self.curr_state]

		self.curr_activity = -1
		self.prev_activity = -1

		self.t_start = 0

		self.initialized = True


	def biphasic_activity(self, t):	  

		# If the number of cycles in the current phase elapsed then switch the phase
		if self.curr_state == 'slow' and self.counter['slow']>=self.N_cycles['slow']:
			self.curr_state = 'fast'
			self.counter['fast'] = 0
			self.t_start = np.copy(t)
		elif self.curr_state =='fast' and self.counter['fast']>=self.N_cycles['fast']:
			self.curr_state = 'slow'
			self.counter['slow'] = 0
			self.t_start = np.copy(t)
			
		# Count each activity cycle in the current phase
		if (self.curr_activity == -1 and self.prev_activity==1) or (self.curr_activity==-1 and self.prev_activity==0):
			self.counter[self.curr_state]+=1
		
		self.prev_activity = self.curr_activity
			
		 # Set the activity parameter based on current state
		self.activity_timescale = self.activity_timescale_biphasic[self.curr_state]
		
		# Get the square-wave activity profile based on current state
		t_elapsed = t - self.t_start # Elapsed time since the start of current activity phase
		
		self.curr_activity = self.square_wave_activity(t_elapsed)
			
		print('{}, {}, {}'.format(t, self.curr_state, self.curr_activity))  

		return self.curr_activity


	def activity_function(self):
		""" Sets the activity function based on activity-profile in the simulation.

			This function is called by the activeFilament object
		"""
		if(self.activity_type == 'square-wave'):
			return lambda t: self.square_wave_activity(t)
		elif(self.activity_type == 'poisson'):
			return lambda t: self.poisson_activity(t)
		elif(self.activity_type == 'normal'):
			return lambda t: self.normal_activity(t)
		elif(self.activity_type == 'biphasic'):
			return lambda t: self.biphasic_activity(t)

	def activity_profile(self, time_array):
		"""Return activity profile for a given time array
		"""
		self.reset_biphasic_activity()
		
		activity_array = np.zeros_like(time_array)

		for ii, t in enumerate(time_array):

			activity_array[ii] = self.activity_function()(t)

		return activity_array


		

