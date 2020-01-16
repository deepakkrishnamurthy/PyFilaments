#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:03:27 2019
Cython class for active filaments
@author: deepak
"""
cimport cython
from scipy.io import savemat
import odespy
from libc.math cimport sqrt, pow
from cython.parallel import prange
import numpy as np
cimport numpy as np
cimport pystokes.unbounded
DTYPE   = np.float
ctypedef np.float_t DTYPE_t

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)


cdef class activeFilament:
    
    def __init__(self, dim_ = 3, Np_ = 10, radius_ = 1, b0_ = 4, k_, eta_, S0_, D0_, shape_, ljeps_):
        
        self.dim = dim_
        self.Np = Np_
        # Particle radius
        self.radius = radius_
        # Equlibrium bond-length
        self.b0 = b0_
        
        # Filament arc-length
        self.L = self.b0*self.Np
        
        
        # Connective spring stiffness
        self.k = k_
        # Bending stiffness
        self.kappa = self.k*self.b0
        
        
        # Fluid viscosity
        self.eta = eta_
        
        # Parameters for the near-field Lennard-Jones potential
        self.ljeps = ljeps_
        self.ljrmin = 2.0*self.radius
        
        # Initial shape of the filament
        self.shape = shape_
        
        # Stresslet strength
        self.S0 = S0_
        
        # Potential-Dipole strength
        self.D0 = D0_
        
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
        
        # Masks for specifying different activities on particles
        # Mask for external forces
        self.F_mag = np.zeros(self.Np*self.dim)
        # Stresslets
        self.S_mag = np.ones(self.Np)
        # Potential dipoles
        self.D_mag = np.zeros(self.Np)
        
        # The most distal particle is motile
#        self.D_mag[-1] = self.D0
        
        # All particle have a stresslet strength of S0
        self.S_mag = self.S_mag*self.S0
        
        
        # Set the colors of the particles based on their activity
        self.particle_colors = []
        self.passive_color = np.reshape(np.array(cmocean.cm.curl(0)),(4,1))
        self.active_color =np.reshape(np.array(cmocean.cm.curl(255)), (4,1))
        
        for ii in range(self.Np):
            
            if(self.S_mag[ii]!=0 or self.D_mag[ii]!=0):
                self.particle_colors.append('r')
            else:
                self.particle_colors.append('b')
                
        print(self.particle_colors)

        
        
        
        # Instantiate the pystokes class
        self.rm = pystokes.unbounded.Rbm(self.radius, self.Np, self.eta)   # instantiate the classes
        # Instantiate the pyforces class
        self.ff = pyforces.forceFields.Forces(self.Np)
        
        # Variables for storing the simulation results
        self.R = None 
        self.Time = None
        
        self.xx = 2*self.Np
        
        self.initializeAll(filament_shape=self.shape)
        
        self.Path = '/Users/deepak/Dropbox/LacryModeling/ModellingResults'
        
        
        self.Folder = 'SimResults_Np_{}_Shape_{}_k_{}_b0_{}_S_{}_D_{}'.format(self.Np, self.shape, self.k, self.b0, self.S0, self.D0)
        
        
        self.saveFolder = os.path.join(self.Path, self.Folder)
        

   
            
    
        
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
            
    def initializeBendingStiffess(self):
        
        # Constant-bending stiffness case
        self.kappa_array = self.kappa*np.ones(self.Np)
        
        # Torque-free ends of the filament
        self.kappa_array[0] = 0
        self.kappa_array[-1] = 0
                
    # Cythonize
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
        
        # rows: dimensions columns : partickes
        # Shape : dim x Np-1
        # Unit separation vectors 
        self.dr_hat = np.vstack((self.dx_hat, self.dy_hat, self.dz_hat))
        
#        print(self.dr_hat)
       
    # Cythonize
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
       
    # Cythonize
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
                vector = self.dr_hat[:,ii-1] + self.dr_hat[:,ii]
                self.t_hat[:,ii] = vector/(np.dot(vector, vector)**(1/2))
                
        
        self.t_hat_array = self.reshapeToArray(self.t_hat)
        
        # Initialize the particle orientations to be along the local tangent vector
        self.p = self.t_hat_array
       
    # Cythonize
    def BendingForces(self):
        # For torque-free filament ends
        
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
    
    # Cythonize
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
                    
#    def initializeActivity(self, F_mask, S_mask, D_mask):
#        # Set the initial activity patter
#        
#        
#          
#    def setActivity(self, t):
##        Apply a specific activity pattern at time t to the filament
                    
    def setForces(self):
        self.F = self.F_mag
      
    
    
    # Cythonize    
    def setStresslet(self):
        
        self.S[:self.Np]            = self.S_mag*(self.p[:self.Np]*self.p[:self.Np] - 1./3)
        self.S[self.Np:2*self.Np]   = self.S_mag*(self.p[self.Np:2*self.Np]*self.p[self.Np:2*self.Np] - 1./3)
        self.S[2*self.Np:3*self.Np] = self.S_mag*(self.p[:self.Np]*self.p[self.Np:2*self.Np])
        self.S[3*self.Np:4*self.Np] = self.S_mag*(self.p[:self.Np]*self.p[2*self.Np:3*self.Np])
        self.S[4*self.Np:5*self.Np] = self.S_mag*(self.p[self.Np:2*self.Np]*self.p[2*self.Np:3*self.Np])
    
    # Cythonize
    def setPotDipole(self):
        
        # Potential dipole axis is along the local orientation vector of the particles
        self.D[:self.Np] = self.D_mag*self.p[:self.Np]
        self.D[self.Np:2*self.Np] = self.D_mag*self.p[self.Np:2*self.Np]
        self.D[2*self.Np:3*self.Np] = self.D_mag*self.p[2*self.Np:3*self.Np]

          
    def initializeAll(self, filament_shape = 'arc'):
        
        
        if(filament_shape == 'line'):
            # Initial particle positions and orientations
            for ii in range(self.Np):
                # The filament is initially linear along x-axis with the first particle at origin
                self.r0[ii] = ii*(self.b0)
                
               
        # Add some Random fluctuations in y-direction
#            self.r0[self.Np:self.xx] = 0.05*self.radius*np.random.rand(self.Np)
        
        elif(filament_shape == 'arc'):
            arc_angle = np.pi

            arc_angle_piece = arc_angle/self.Np
            
            for ii in range(self.Np):
                # The filament is initially linear along x-axis with the first particle at origin
                if(ii==0):
                    self.r0[ii], self.r0[ii+self.Np], self.r0[ii+self.xx] = 0,0,0 
                else:
                    self.r0[ii] = self.r0[ii-1] + self.b0*np.sin(ii*arc_angle_piece)
                    self.r0[ii + self.Np] = self.r0[ii-1 + self.Np] + self.b0*np.cos(ii*arc_angle_piece)
                    
                
        elif(filament_shape == 'sinusoid'):
            nWaves = 2.5
            Amp=3
            for ii in range(self.Np):
                # The filament is initially linear along x-axis with the first particle at origin
                self.r0[ii] = ii*(self.b0)
                self.r0[ii + self.Np] = Amp*np.sin(nWaves*self.r0[ii]*2*np.pi/((self.Np-1)*self.b0))
            
            
            
        
        
        self.r = self.r0
        
        
        # Initialize the bending-stiffness array
        self.initializeBendingStiffess()
        self.getSeparationVector()
        self.getBondAngles()
        
        self.getTangentVectors()
        # Orientation vectors of particles depend on local tangent vector
        self.p0 = self.p
     
    # Cythonize
    def calcForces(self):
        
        self.F = self.F*0
        
     
        self.ff.lennardJones(self.F, self.r, self.ljeps, self.ljrmin)
        self.ConnectionForces()
        self.BendingForces()
        # Add all the intrinsic forces together
        self.F += self.F_conn + self.F_bending_array


        
        # Add external forces
#        self.ff.sedimentation(self.F, g = -10)
        
        
    # Cythonize   
    def rhs(self, r, t):
        
        self.drdt = self.drdt*0
        
        self.r = r
        
        
        self.getSeparationVector()
        self.getBondAngles()
        self.getTangentVectors()
        
        self.setStresslet()
        
        self.calcForces()
        
#        print(self.F)
        
        # Stokeslet contribution to Rigid-Body-Motion
        self.rm.stokesletV(self.drdt, self.r, self.F)
        
        # Stressslet contribution to Rigid-Body-Motion
        self.rm.stressletV(self.drdt, self.r, self.S)
        
#        self.rm.potDipoleV(self.drdt, self.r, self.p)
        
        # Apply the boundary condition (1st particle is fixed in space)
#        self.drdt[0], self.drdt[self.Np], self.drdt[self.xx] = 0,0,0 
        
        
    
    def simulate(self, Tf, Npts, save = False, overwrite = False):
        
        self.saveFile = 'SimResults_Tmax_{}_Np_{}_Shape_{}_S_{}_D_{}.pkl'.format(Tf, self.Np, self.shape, self.S0, self.D0)
        if(save):
            if(not os.path.exists(self.saveFolder)):
                os.makedirs(self.saveFolder)
        
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
                self.saveResults()
                
        else:
            print('Loading Simulation from disk ....')
            with open(os.path.join(self.saveFolder, self.saveFile), 'rb') as f:
            
                self.Np, self.b0, self.k, self.S0, self.D0, self.R, self.Time = pickle.load(f)
            
        
        
#        savemat('Np=%s_vs=%4.4f_K=%4.4f_s_0=%4.4f.mat'%(self.Np, self.vs, self.k, self.S0), {'X':u, 't':t, 'Np':self.Np,'k':self.k, 'vs':self.vs, 'S0':self.S0,})
        
    
    def saveResults(self):
        
        if(self.R is not None):
            
            
            with open(os.path.join(self.saveFolder, self.saveFile), 'wb') as f:
                pickle.dump((self.Np, self.b0, self.k, self.S0, self.D0, self.R, self.Time), f)
                
                
            
#        
            
            
        
    
        









