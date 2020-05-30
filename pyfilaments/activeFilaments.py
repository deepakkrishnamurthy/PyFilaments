#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:35:55 2019
-------------------------------------------------------------------------------
> Simulate extensible, elastic filaments using active colloid theory.
> Filaments are made up of discrete active colloids.
> Full hydrodynamic interactions to the order necessary to solve for rigid body motions is solved 
using the PyStokes library (R Singh et al ...).
> Non-linear springs provide connectivity.
> Bending is penalized as an elastic potential.
> Nearfield repulsion using a Lennard-Jones-based potential
> Dynamic time-based activity profiles. 
-------------------------------------------------------------------------------
@author: deepak
"""
from __future__ import division
import pystokes
import pyforces
import numpy as np
import odespy
import os
import cmocean
import pickle
import matplotlib.pyplot as plt 

from scipy import signal
from scipy import interpolate

import h5py


class activeFilament:
	'''
		This is the main active Filament class that calls the pyStokes and pyForces libraries 
		for solving hydrodynamic and steric interactions.
	'''
	def __init__(self, dim = 3, Np = 3, radius = 1, b0 = 1, k = 1, mu = 1.0/6, F0 = 0, S0 = 0, D0 = 0, 
					shape ='line', scale_factor = None, bc = {0:'clamped', -1:'free'}):
		
		#-----------------------------------------------------------------------------
		# Filament parameters
		#-----------------------------------------------------------------------------
		self.dim = dim

		# IC: Initial Conditions: Initial shape of the filament
		self.shape = shape

		# BC: Boundary conditions:
		self.bc = bc

		# Sets particle number based on BC. Each Clamped BC increase particle number by 1.
		self.setParticleNumber(Np = Np)

		# Particle radius
		self.radius = radius
		# Equlibrium bond-length
		self.b0 = b0
		
		# Filament arc-length
		self.L = self.b0*self.Np
		
		
		# Connective spring stiffness
		self.k = k
		
		# Bending stiffness
		# self.kappa_hat = self.k*self.b0
		
		# 30 May 2020: Important change. The bending stiffness and axial stiffness now are for a homogeneous elastic rod.
		self.kappa_hat = ((self.radius**2)/4)*self.k
		
		# Fluid viscosity
		self.mu = mu
		
		# Parameters for the near-field Lennard-Jones potential
		self.ljeps = 0.1
		self.ljrmin = 2.0*self.radius
		

		# Body-force strength
		self.F0 = F0

		# Stresslet strength
		self.S0 = S0
		
		# Potential-Dipole strength
		self.D0 = D0

		# Simulation type
		self.sim_type = None
			
		
		# Instantiate the pystokes class
		self.rm = pystokes.unbounded.Rbm(self.radius, self.Np, self.mu)   # instantiate the classes
		# Instantiate the pyforces class
		self.ff = pyforces.forceFields.Forces(self.Np)
		
		# Initialize arrays for storing particle positions, activity strengths etc.
		self.allocate_arrays()

		self.initialize_filament()

		self.setParticleColors()
		
		
	def allocate_arrays(self):
		# Initiate positions, orientations, forces etc of the particles
		self.r = np.zeros(self.Np*self.dim)
		self.p = np.zeros(self.Np*self.dim)
		
		self.r0 = np.zeros(self.Np*self.dim)
		self.p0 = np.zeros(self.Np*self.dim)
		
		# Velocity of all the particles
		self.drdt = np.zeros(self.Np*self.dim)
		
		self.F = np.zeros(self.Np*self.dim)


		self.T = np.zeros(self.Np*self.dim)
		# Stresslet 
		self.S = np.zeros(5*self.Np)
		# Potential dipole
		self.D = np.zeros(self.Np*self.dim)
		
		# Masks for specifying different activities on particles
		# Mask for external forces
		self.F_mag = np.zeros(self.Np*self.dim)
		# Stresslets
		self.S_mag = np.zeros(self.Np)
		# Potential dipoles
		self.D_mag = np.zeros(self.Np)

		# Variables for storing the simulation results
		self.R = None 
		self.Time = None
   
	def setParticleNumber(self, Np = 3):
		# Sets the number of simulated particles based on BC. 
		# For clamped BC the number of particles is one extra for each clamped BC to implement the BC using a dummy particle.
		count = 0
		for key in self.bc:
			if(self.bc[key] == 'clamped'):
				count += 1

		self.Np = Np + count

		self.xx = 2*self.Np


	# Set the colors of the particles based on their activity
	def setParticleColors(self):
		self.particle_colors = []
		self.passive_color = np.reshape(np.array(cmocean.cm.curl(0)),(4,1))
		self.active_color =np.reshape(np.array(cmocean.cm.curl(255)), (4,1))
		
		for ii in range(self.Np):
			
			if(self.S_mag[ii]!=0 or self.D_mag[ii]!=0):
				self.particle_colors.append('r')
			else:
				self.particle_colors.append('b')
				
		print(self.particle_colors)
	
		
	def reshapeToArray(self, Matrix):
		# Takes a matrix of shape (dim, Np) and reshapes to an array (dim*Np, 1) 
		# where the convention is [x1, x2 , x3 ... X_Np, y1, y2, .... y_Np, z1, z2, .... z_Np]
		nrows, ncols = np.shape(Matrix)
		return np.squeeze(np.reshape(Matrix, (nrows*ncols,1), order = 'C'))
	
	def reshapeToMatrix(self, Array):
		# Takes an array of shape (dim*N, 1) and reshapes to a Matrix  of shape (dim, N) and 
		# where the array convention is [x1, x2 , x3 ... X_Np, y1, y2, .... y_Np, z1, z2, .... z_Np]
		# and matrix convention is |x1 x2 ...  |
		#                          |y1 y2 ...  |
		#                          |z1 z2 ...  |
		array_len = len(Array)
		ncols = array_len/self.dim
		return np.reshape(Array, (self.dim, ncols), order = 'C')
			
	
				
	# calculate the pair-wise separation vector
	def getSeparationVector(self):
		
	 
		# length Np-1
		self.dx = self.r[1:self.Np] - self.r[0:self.Np-1]
		self.dy = self.r[self.Np+1:2*self.Np] - self.r[self.Np:2*self.Np-1]
		self.dz = self.r[2*self.Np+1:3*self.Np] - self.r[2*self.Np:3*self.Np-1]
		
		
		# length Np-1
		# Lengths of the separation vectors
		self.dr = (self.dx**2 + self.dy**2 + self.dz**2)**(1/2)
		
		# length Np-1
		self.dx_hat = self.dx/self.dr
		self.dy_hat = self.dy/self.dr
		self.dz_hat = self.dz/self.dr
		
		# rows: dimensions, columns : particles
		# Shape : dim x Np-1
		# Unit separation vectors 
		self.dr_hat = np.vstack((self.dx_hat, self.dy_hat, self.dz_hat))
		
#        print(self.dr_hat)
		
	# Calculate bond-angle vector for the filament
	def getBondAngles(self):
		
		self.getSeparationVector()
		
		# The number of angles equals the no:of particles
		self.cosAngle = np.zeros(self.Np)
		
		
		self.startAngle = 0
		self.endAngle = 0
		
		for ii in range(self.Np-1):
			
			# For the boundary-points, store the angle wrt to the x-axis of the global cartesian coordinate system
			if(ii==0 or ii == self.Np-1):
				
				self.cosAngle[ii] = np.dot(self.dr_hat[:,ii], [1, 0 , 0])
				
			else:
				self.cosAngle[ii] = np.dot(self.dr_hat[:,ii-1], self.dr_hat[:,ii] )
				
		
#        print(self.cosAngle)
		
	# Find the local tangent vector of the filament at the position of each particle
	def getTangentVectors(self):
		
		# Unit tangent vector at the particle locations
		self.t_hat = np.zeros((self.dim,self.Np))
		
		for ii in range(self.Np):
			
			if ii==0:
				self.t_hat[:,ii] = self.dr_hat[:,ii]
			elif ii==self.Np-1:
				self.t_hat[:,-1] = self.dr_hat[:,-1]
			else:
				# For particles that have a neighbhor on each side, the tangent vector is the average of the two bonds. 
				vector = self.dr_hat[:,ii-1] + self.dr_hat[:,ii]
				self.t_hat[:,ii] = vector/(np.dot(vector, vector)**(1/2))
				
		
		self.t_hat_array = self.reshapeToArray(self.t_hat)
		
		# Initialize the particle orientations to be along the local tangent vector
		self.p = self.t_hat_array
				
	


	def BendingForces(self):
		# See notes for derivation of these expressions for bending potentials.

		self.getBondAngles()
		
		self.F_bending = np.zeros((self.dim,self.Np))
		
		for ii in range(self.Np):
			
			if ii==0:
				# Torque-free ends
				self.F_bending[:,ii] = self.kappa_array[ii+1]*(1/self.dr[ii])*(self.dr_hat[:, ii]*self.cosAngle[ii+1] - self.dr_hat[:, ii+1])
				
			elif ii == self.Np-1:
				# Torque-free ends
				self.F_bending[:,ii] = self.kappa_array[ii-1]*(1/self.dr[ii-1])*(self.dr_hat[:, ii-2] - self.cosAngle[ii - 1]*self.dr_hat[:, ii-1])
				
			else:
				
				if(ii!=1):
					term_n_minus = self.kappa_array[ii-1]*(self.dr_hat[:, ii-2] - self.dr_hat[:, ii-1]*self.cosAngle[ii-1])*(1/self.dr[ii-1])
				else:
					term_n_minus = 0
					
				term_n1 = (1/self.dr[ii-1] + self.cosAngle[ii]/self.dr[ii])*self.dr_hat[:, ii]
				term_n2 = -(1/self.dr[ii] + self.cosAngle[ii]/self.dr[ii-1])*self.dr_hat[:, ii-1]
				
				term_n = self.kappa_array[ii]*(term_n1 + term_n2)
				
				if(ii!=self.Np-2):
					term_n_plus = self.kappa_array[ii+1]*(-self.dr_hat[:, ii+1] + self.dr_hat[:, ii]*self.cosAngle[ii + 1])*(1/self.dr[ii])
				else:
					term_n_plus = 0
					
				self.F_bending[:,ii] =  term_n_minus + term_n + term_n_plus
				
			
		# Now reshape the forces array
		self.F_bending_array = self.reshapeToArray(self.F_bending)  
 
	
	def ConnectionForces(self):
	
#        def int Np = self.Np, i, j, xx = 2*Np
#        def double dx, dy, dz, dr2, dr, idr, fx, fy, fz, fac
		xx = 2*self.Np
		self.F_conn = np.zeros(self.dim*self.Np)
		
		for i in range(self.Np):
			fx = 0.0; fy = 0.0; fz = 0.0;
			for j in range(i,self.Np):
				
				if((i-j)==1 or (i-j)==-1):
					
					dx = self.r[i   ] - self.r[j   ]
					dy = self.r[i+self.Np] - self.r[j+self.Np]
					dz = self.r[i+xx] - self.r[j+xx] 
					dr2 = dx*dx + dy*dy + dz*dz
					dr = dr2**(1/2)
					
	#                    dr_hat = np.array([dx, dy, dz], dtype = 'float')*(1/dr)
					
					fac = -self.k*(dr - self.b0)
				
					fx = fac*dx/dr
					fy = fac*dy/dr
					fz = fac*dz/dr
					
					
					# Force on particle "i"
					self.F_conn[i]    += fx 
					self.F_conn[i+self.Np] += fy 
					self.F_conn[i+xx] += fz 
					
					# Force on particle "j"
					self.F_conn[j]    -= fx 
					self.F_conn[j+self.Np] -= fy 
					self.F_conn[j+xx] -= fz 

		
		
	def setStresslet(self):
		# Specifies the stresslet on each active particle
		
		self.S[:self.Np]            = self.S_mag*(self.p[:self.Np]*self.p[:self.Np] - 1./3)
		self.S[self.Np:2*self.Np]   = self.S_mag*(self.p[self.Np:2*self.Np]*self.p[self.Np:2*self.Np] - 1./3)
		self.S[2*self.Np:3*self.Np] = self.S_mag*(self.p[:self.Np]*self.p[self.Np:2*self.Np])
		self.S[3*self.Np:4*self.Np] = self.S_mag*(self.p[:self.Np]*self.p[2*self.Np:3*self.Np])
		self.S[4*self.Np:5*self.Np] = self.S_mag*(self.p[self.Np:2*self.Np]*self.p[2*self.Np:3*self.Np])
	
	def setPotDipole(self):
		# Specifies the potential dipole on each active particle
		
		# Potential dipole axis is along the local orientation vector of the particles
		self.D[:self.Np] = self.D_mag*self.p[:self.Np]
		self.D[self.Np:2*self.Np] = self.D_mag*self.p[self.Np:2*self.Np]
		self.D[2*self.Np:3*self.Np] = self.D_mag*self.p[2*self.Np:3*self.Np]

	
	def initialize_filamentShape(self):


		if(self.shape == 'line'):
			# Initial particle positions and orientations
			for ii in range(self.Np):
				# The filament is initially linear along x-axis with the first particle at origin
				self.r0[ii] = ii*(self.b0)
				
			   
		# Add some Random fluctuations in y-direction
#            self.r0[self.Np:self.xx] = 0.05*self.radius*np.random.rand(self.Np)
		
		elif(self.shape == 'arc'):
			arc_angle = np.pi

			arc_angle_piece = arc_angle/self.Np
			
			for ii in range(self.Np):
				# The filament is initially linear along x-axis with the first particle at origin
				if(ii==0):
					self.r0[ii], self.r0[ii+self.Np], self.r0[ii+self.xx] = 0,0,0 

				else:
					self.r0[ii] = self.r0[ii-1] + self.b0*np.sin(ii*arc_angle_piece)
					self.r0[ii + self.Np] = self.r0[ii-1 + self.Np] + self.b0*np.cos(ii*arc_angle_piece)
					
				
		elif(self.shape == 'sinusoid'):
			nWaves = 1
			Amp = 1e-4
			for ii in range(self.Np):

				# The first particle at origin
				if(ii==0):
					self.r0[ii], self.r0[ii+self.Np], self.r0[ii+self.xx] = 0,0,0 

				else:
					self.r0[ii] = ii*(self.b0)
					self.r0[ii + self.Np] = Amp*np.sin(nWaves*self.r0[ii]*2*np.pi/((self.Np-1)*self.b0))
		
		# Apply the kinematic boundary conditions to the filament ends

		self.ApplyBC_position()

		self.r = self.r0


	def initializeBendingStiffess(self):
		# bc: dictionary that holds the boundary-conditions at the two ends of the filaments 
		# Constant-bending stiffness case
		self.kappa_array = self.kappa_hat*np.ones(self.Np)
		
		for key in self.bc:
			value = self.bc[key]

			if value=='free' or value == 'fixed':
				# The bending stiffness is set to zero only for 'free' or 'fixed' boundary conditions
				print('Assigning {} BC to filament end {}'.format(value, key))
				self.kappa_array[key] = 0 

		print(self.kappa_array)

			  
	def initialize_filament(self):
		
		
		self.initialize_filamentShape()
		
		# Initialize the bending-stiffness array
		self.initializeBendingStiffess()
		self.getSeparationVector()
		self.getBondAngles()
		
		self.getTangentVectors()
		# Orientation vectors of particles depend on local tangent vector
		self.p0 = self.p
		
	def internal_forces(self):
		
		self.F = self.F*0
		
	 
		self.ff.lennardJones(self.F, self.r, self.ljeps, self.ljrmin)
		
		# Print out the lennardJones forces
		# print(self.F)

		self.ConnectionForces()
		self.BendingForces()
		# Add all the intrinsic forces together
		self.F += self.F_conn + self.F_bending_array


		
		# Add external forces
#        self.ff.sedimentation(self.F, g = -10)

	def external_forces(self):

		self.F += self.F_mag

	def ApplyBC_position(self):
		'''
		Apply the kinematic boundary conditions:
		'''
		for key in self.bc:

			bc_value = self.bc[key]

			# Proximal end
			if(key == 0):
				# Index corresponding to end particle and next nearest particle (proximal end)
				end = 0
				end_1 = 1
				pos_end = (0,0,0)
				pos_end_1 = (self.b0,0,0)
				
			# Distal end
			elif(key == -1 or key == self.Np-1):
				# Index correspond to end particle and next nearest particle (distal end)
				end = self.Np - 1
				end_1 = self.Np - 2

				pos_end_1 = ((self.Np - 2)*self.b0,0,0)
				pos_end = ((self.Np - 1)*self.b0,0,0)
				

			if(bc_value == 'fixed'):
				
				self.r0[end], self.r0[end + self.Np], self.r0[end + self.xx]  = pos_end

			elif(bc_value == 'clamped'):

				self.r0[end], self.r0[end + self.Np], self.r0[end + self.xx] = pos_end

				self.r0[end_1], self.r0[end_1 + self.Np], self.r0[end_1 + self.xx] = pos_end_1



	def ApplyBC_velocity(self):
		'''
		Apply the kinematic boundary conditions as a velocity condition:
		'''
		
		for key in self.bc:

			bc_value = self.bc[key]

			# Proximal end
			if(key==0):
				# Index correspond to end particle and next nearest particle (proximal end)
				end = 0
				end_1 = 1

				vel_end = (0,0,0)
				vel_end_1 = (0,0,0)
				
			# Distal end
			elif(key == -1 or key == self.Np-1):
				# Index correspond to end particle and next nearest particle (distal end)
				end = self.Np - 1
				end_1 = self.Np - 2

				vel_end_1 = (0,0,0)
				vel_end = (0,0,0)
				

			if(bc_value == 'fixed'):
				self.drdt[end], self.drdt[end + self.Np], self.drdt[end + self.xx]  = vel_end
			elif(bc_value == 'clamped'):

				# Apply velocity bc to the farthermost particle
				self.drdt[end], self.drdt[end + self.Np], self.drdt[end + self.xx]  = vel_end

				# Apply velocity bc to the next to the farthermost particle

				self.drdt[end_1], self.drdt[end_1 + self.Np], self.drdt[end_1 + self.xx]  = vel_end_1

	def setActivityForces(self, t):

		if(self.sim_type == 'point'):
			'''Simulates active filament where only the distal particle has time-dependent activity.
			'''
			self.D_mag[-1] = self.D0*self.activity_profile(t)

		elif(self.sim_type == 'dist'):
			'''
			Simulates active filament where the activity pattern models that in Lacrymaria olor.
			Distal particle: 
				1. Active during extension.
				2. Stalled during reversal.
			All other particles: 
				1. Inactive during extension.
				2. Active during reversal. 

			Scale factor: 
				Quantifies the relative strengths of the Distal particle vs Other particles activity.
			'''

			if(self.activity_profile(t)==1):
		
				# self.D_mag[:self.Np-1] = self.D0/self.scale_factor
				self.D_mag[-1] = self.D0

			elif(self.activity_profile(t)==-1):
				self.D_mag[:self.Np-1] = -self.D0/self.scale_factor
				self.D_mag[-1] = 0

		

		
	def rhs(self, r, t):
		
		self.setActivityForces(t = t)
	
		self.drdt = self.drdt*0
		
		self.r = r
		
		
		self.getSeparationVector()
		self.getBondAngles()
		self.getTangentVectors()

		
		self.setStresslet()

		self.setPotDipole()
		
		self.internal_forces()

		self.external_forces()
				
		# Stokeslet contribution to Rigid-Body-Motion
		# This is equivalent to calculating the RBM due to a stokeslet component of the active colloid.
		self.rm.stokesletV(self.drdt, self.r, self.F)
		
		# Stresslet contribution to Rigid-Body-Motion
		# @@@ (TO DO) For efficiency calculate this Only if any element of the Stresselt strength is non-zero
		
		if(self.sim_type != 'sedimentation'):
			# For sedimentation the stresslet and potDipole contribution is zero.
			self.rm.stressletV(self.drdt, self.r, self.S)
			
			self.rm.potDipoleV(self.drdt, self.r, self.D)
		
		# Apply the kinematic boundary conditions as a velocity condition
		self.ApplyBC_velocity()
		
		
	
	def simulate(self, Tf = 100, Npts = 10, sim_type = 'point', activity_profile = None, scale_factor = 1, 
				activity_timescale = 0, save = False, path = '/Users/deepak/Dropbox/LacryModeling/ModellingResults',overwrite = False):
		
		# Set the simulation type
		self.sim_type = sim_type

		self.activity_timescale = activity_timescale

		# Set the scale-factor
		self.scale_factor = scale_factor
		#---------------------------------------------------------------------------------
		#Allocate a Path and folder to save the results


		
		self.path = path

		if(not os.path.exists(self.path)):
			os.makedirs(self.path)

		self.folder = 'SimResults_Np_{}_Shape_{}_k_{}_b0_{}_F_{}_S_{}_D_{}_scalefactor_{}_{}'.format\
							(self.Np, self.shape, self.k, self.b0, self.F0, self.S0, self.D0, 
							int(self.activity_timescale), self.scale_factor, sim_type)

		self.saveFolder = os.path.join(self.path, self.folder)


		self.saveFile = 'SimResults_Np_{}_Shape_{}_k_{}_b0_{}_F_{}_S_{}_D_{}_actTime_{}_scaleFactor_{}_{}.hdf5'.format\
							(self.Np, self.shape, self.k, self.b0, self.F0, self.S0, self.D0, 
							int(self.activity_timescale), self.scale_factor, sim_type)

		if(save):
			if(not os.path.exists(self.saveFolder)):
				os.makedirs(self.saveFolder)
		#---------------------------------------------------------------------------------

		# Set the activity profile
		self.activity_profile = activity_profile

		# if simulating constant external forces
		if(self.sim_type == 'sedimentation'):
			''' Simulates a filament where there is a net body force on each particle making up the filament.
			'''

			Np = self.Np
			xx = 2*Np 

			for ii in range(Np):
					# F[i   ] +=  0
					self.F_mag[ii+Np] += self.F0
				# F[i+xx] +=  g

			self.S_mag[:] = 0
			self.D_mag[:] = 0


		#---------------------------------------------------------------------------------

		def rhs0(r, t):
			# Pass the current time from the ode-solver, 
			# so as to implement time-varying conditions
			self.rhs(r, t)
			print(t)
			return self.drdt
		
		if(not os.path.exists(os.path.join(self.saveFolder, self.saveFile)) or overwrite==True):
			print('Running the filament simulation ....')
			# integrate the resulting equation using odespy
			T, N = Tf, Npts;  time_points = np.linspace(0, T, N+1);  ## intervals at which output is returned by integrator. 
			solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
			solver.set_initial_condition(self.r0)
			self.R, self.Time = solver.solve(time_points)
			
			if(save):
				print('Saving results...')
				self.save_data()
				
		else:
			self.load_data(os.path.join(self.saveFolder, "simresults.hdf5"))

						
	# def loadData(self, File):
	# 	print('Loading Simulation from disk ....')

	# 	with open(File, 'rb') as f:
			
	# 		self.Np, self.b0, self.k, self.S0, self.D0, self.F_mag, self.S_mag, self.D_mag, self.R, self.Time = pickle.load(f)
	
	# def saveData(self):
		
	# 	if(self.R is not None):
			
			
	# 		with open(os.path.join(self.saveFolder, self.saveFile), 'wb') as f:
	# 			pickle.dump((self.Np, self.b0, self.k, self.S0, self.D0, self.F_mag, self.S_mag, self.D_mag, self.R, self.Time), f)

	def load_data(self, file = None):

		print('Loading Simulation data from disk ...')

		if(file is not None):

			if(file[-4:] == 'hdf5'):

				with h5py.File(file, "r") as f:

					# Load the simulation data
					self.Time = f["Time"][:]
					dset = f["Position"]
					self.R = dset[:]


					# Load the metadata:
					self.Np = dset.attrs['N particles']
					self.radius = dset.attrs['radius']
					self.b0 = dset.attrs['bond length']
					self.k = dset.attrs['spring constant'] 

					self.kappa_hat = dset.attrs['kappa_hat']

					self.F0 = dset.attrs['force strength']

					self.S0 = dset.attrs['stresslet strength'] 
					self.D0 = dset.attrs['potDipole strength']

					self.activity_timescale = dset.attrs['activity time scale']

					self.sim_type = dset.attrs['simulation type']

					self.F_mag = f["particle forces"][:]
					
					self.S_mag = f["particle stresslets"][:]

					self.D_mag = f["particle potDipoles"][:]

					self.activity_profile = f["activity profile"][:]

					

			else:
				with open(file, 'rb') as f:
			
					self.Np, self.b0, self.k, self.S0, self.D0, self.F_mag, self.S_mag, self.D_mag, self.R, self.Time = pickle.load(f)







	# Implement a save module based on HDF5 format:
	def save_data(self):


		with h5py.File(os.path.join(self.saveFolder, self.saveFile), "w") as f:

			f.create_dataset("Time", data = self.Time)

			# Position contains the 3D positions of the Np particles over time. 
			dset = f.create_dataset("Position", data = self.R)

			dset.attrs['N particles'] = self.Np
			dset.attrs['radius'] = self.radius
			dset.attrs['bond length'] = self.b0
			dset.attrs['spring constant'] = self.k
			dset.attrs['kappa_hat'] = self.kappa_hat
			dset.attrs['force strength'] = self.F0
			dset.attrs['stresslet strength'] = self.S0
			dset.attrs['potDipole strength'] = self.D0
			dset.attrs['simulation type'] = self.sim_type
			dset.attrs['activity time scale'] = self.activity_timescale


			
			f.create_dataset("particle forces", data = self.F_mag)
			f.create_dataset("particle stresslets", data = self.S_mag)
			f.create_dataset("particle potDipoles", data = self.D_mag)

			if(self.activity_profile is not None):
				f.create_dataset("activity profile", data = self.activity_profile)







				   
	def plotFilament(self, r = None):
		
	
		ax1 = plt.gca()
		
#        1ax = fig.add_subplot(1,1,1)
		

		ax1.scatter(r[:self.Np], r[self.Np:2*self.Np], 20, c = self.particle_colors, alpha = 0.75, zorder = 20, cmap = cmocean.cm.curl)
		ax1.plot(r[:self.Np], r[self.Np:2*self.Np], color = 'k', alpha = 0.5, zorder = 10)

#        ax.set_xlim([-0.1, self.Np*self.b0])
#        ax.set_ylim([-self.Np*self.b0/2, self.Np*self.b0/2])
		
#        fig.canvas.draw()

			
		
		












