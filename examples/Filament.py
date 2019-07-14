#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:35:55 2019
-------------------------------------------------------------------------------
*Simulate extensile, elastic filaments using active colloid theory.
*Filaments are made up of discrete active colloids.
*Full hydrodynamic interactions to the order necessary to fully solve for rigid body motions is solved 
using the PyStokes library (R Singh et al ...).
*Non-linear springs provide connectivity.
*Bending is penalized as an elastic potential.
*Nearfield reupulsion using a Lennard-Jones-based potential
-------------------------------------------------------------------------------
@author: deepak
"""
from __future__ import division
import pystokes
import pyforces
import matplotlib.pyplot as plt 
import numpy as np

class activeFilament:
    
    def __init__(self, dim = 3, Np = 1, radius = 1, b0 = 2, k = 1, kappa = 1):
        
        self.dim = dim
        self.Np = Np
        # Particle radius
        self.radius = radius
        # Equlibrium bond-length
        self.b0 = b0
        # Connective spring stiffness
        self.k = k
        # Bending stiffness
        self.kappa = kappa
        
        # Initiate positions, orientations, forces etc of the particles
        self.r = np.zeros(self.Np*self.dim)
        self.p = np.zeros(self.Np*self.dim)
        
        self.F = np.zeros(self.Np*self.dim)
        self.T = np.zeros(self.Np*self.dim)
        # Stresslet strength
        self.S = np.zeros(self.Np*self.dim)
        
        
        
        self.initializeFilament()
        self.initializeBendingStiffess();
        
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
        
    def initializeFilament(self):
        
        for ii in range(self.Np):
            # The filament is initially linear along x-axis with the first particle at origin
            self.r[ii] = ii*self.b0
            self.p[ii] = 1
           
        # Random fluctuations in y-direction
        self.r[self.Np:self.Np*2] = np.random.rand(self.Np)
            
    def initializeBendingStiffess(self):
        
        # Constant-bending stiffness case
        self.kappa_array = self.kappa*np.ones(self.Np)
        
        # Torque-free ends of the filament
        self.kappa_array[0] = 0
        self.kappa_array[-1] = 0
        
    def plotFilament(self):
        
    
        fig = plt.figure()
        
        ax = fig.add_subplot(1,1,1)
        

        ax.scatter(self.r[:self.Np], self.r[self.Np:2*self.Np], 200, color = 'b', alpha = 1.0, zorder = 2)
        ax.plot(self.r[:self.Np], self.r[self.Np:2*self.Np], color = 'k', alpha = 0.5, zorder = 1)

        ax.set_xlim([-0.1, self.Np*self.b0])
        ax.set_ylim([-self.Np*self.b0/2, self.Np*self.b0/2])
        
        fig.canvas.draw()
        
    # calculate the pair-wise separation vector
    def getSeparationVector(self):
        
     
        # length Np-1
        self.dx = self.r[1:self.Np] - self.r[0:self.Np-1]
        self.dy = self.r[self.Np+1:2*self.Np] - self.r[self.Np:2*self.Np-1]
        self.dz = self.r[2*self.Np+1:3*self.Np] - self.r[2*self.Np:3*self.Np-1]
        
        # length Np-1
        self.dr = (self.dx**2 + self.dy**2 + self.dz**2)**(1/2)
        
        # length Np-1
        self.dx_hat = self.dx/self.dr
        self.dy_hat = self.dy/self.dr
        self.dz_hat = self.dz/self.dr
        
        # rows: dimensions columns : partickes
        # Shape : dim x Np-1
        self.dr_hat = np.vstack((self.dx_hat, self.dy_hat, self.dz_hat))
        
        print(self.dr_hat)
        
        
        
        
        
    
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
                
        
        print(self.cosAngle)
        
        
    def findBendingForces(self):
        # For torque-free filament ends
        
        self.getBondAngles()
        
        F_bending = np.zeros((self.dim,self.Np))
        
        for ii in range(self.Np):
            
            if ii==0:
                # Torque-free ends
                F_bending[:,ii] = self.kappa_array[ii+1]*(1/self.dr[ii])*(self.dr_hat[:, ii]*self.cosAngle[ii+1] - self.dr_hat[:, ii+1])
                
            elif ii == self.Np-1:
                # Torque-free ends
                F_bending[:,ii] = self.kappa_array[ii-1]*(1/self.dr[ii-1])*(self.dr_hat[:, ii-2] - self.cosAngle[ii]*self.dr_hat[:, ii-1])
                
            else:
                
                if(ii!=1):
                    term_n_minus = self.kappa_array[ii-1]*(self.dr_hat[:, ii-2] - self.dr_hat[:, ii-1]*self.cosAngle[ii-1])*(1/self.dr[ii-1])
                else:
                    term_n_minus = 0
                    
                term_n1 = self.kappa_array[ii]*((1/self.dr[ii-1] + self.cosAngle[ii]/self.dr[ii])*self.dr_hat[:, ii])
                term_n2 = -self.kappa_array[ii]*((1/self.dr[ii] + self.cosAngle[ii]/self.dr[ii-1])*self.dr_hat[:, ii-1])
                
                term_n = term_n1 + term_n2
                
                if(ii!=self.Np-2):
                    term_n_plus = self.kappa_array[ii+1]*(-self.dr_hat[:, ii+1] + self.dr_hat[ii]*self.cosAngle[ii + 1])*(1/self.dr[:, ii])
                else:
                    term_n_plus = 0
                    
                F_bending[:,ii] =  term_n_minus + term_n + term_n_plus
                
            
        # Now reshape the forces array
        
        return self.reshapeToArray(F_bending)
                
                
            
        
            
            
#def cross(v1, v2, Np = 1, dim = 3):
#    v1, v2 = np.array(v1), np.array(v2)
#    
#    V1, V2 = np.zeros((dim, Np)),np.zeros((dim, Np))
#    
#    for ii in range(Np):
#        for jj in range(dim):
#            V1[jj, ii] = v1[ii + jj]
#            V2[jj, ii] = v2[ii + jj]
    
#-------------------------------------------------------------------------------
def connectivity(r, kappa = 1,b0 = 1):
    
#        def int Np = self.Np, i, j, xx = 2*Np
#        def double dx, dy, dz, dr2, dr, idr, fx, fy, fz, fac
    xx = 2*Np
    F = np.zeros(dim*Np)
    
    for i in range(Np):
        fx = 0.0; fy = 0.0; fz = 0.0;
        for j in range(i,Np):
            
            if((i-j)==1 or (i-j)==-1):
                
                dx = r[i   ] - r[j   ]
                dy = r[i+Np] - r[j+Np]
                dz = r[i+xx] - r[j+xx] 
                dr2 = dx*dx + dy*dy + dz*dz
                dr = dr2**(1/2)
                
#                    dr_hat = np.array([dx, dy, dz], dtype = 'float')*(1/dr)
                
                fac = -kappa*(dr - b0)
            
                fx = fac*dx/dr
                fy = fac*dy/dr
                fz = fac*dz/dr
                
                print(fx)
                
                # Force on particle "i"
                F[i]    += fx 
                F[i+Np] += fy 
                F[i+xx] += fz 
                
                # Force on particle "j"
                F[j]    -= fx 
                F[j+Np] -= fy 
                F[j+xx] -= fz 
        
        
    return F

#-------------------------------------------------------------------------------
# Simulation Parameters
#-------------------------------------------------------------------------------
a,  Np = 1, 2            # radius and number of particles
L, dim = 128,  3                    # size and dimensionality and the box
v = np.zeros(dim*Np)                # Memory allocation for velocity
omega = np.zeros(dim*Np)                # Memory allocation for angular velocities

r = (L/2)*np.ones(dim*Np)                # Position vector of the particles
p = np.zeros(dim*Np)                # Position vector of the particles
S = np.zeros(5*Np)                # Forces on the particles
F = np.zeros(dim*Np)                 # Forces on the particles
F_sed = np.zeros(dim*Np)                 # Forces on the particles
F_conn = np.zeros(dim*Np)                 # Forces on the particles
#-------------------------------------------------------------------------------

Nb, Nm = 1, 4

# Connective spring stiffness
kappa = float(5)
# Equlibrium bond-length
b0 = float(2*a)

# Initialize the particle positions and orientations
r[5] = r[5]-4*a

p[2*Np:3*Np] = 1

#p[0] = 1
#p[1] = -1

S0 = 5
S[:Np]      = S0*(p[:Np]*p[:Np] - 1./3)
S[Np:2*Np]  = S0*(p[Np:2*Np]*p[Np:2*Np] - 1./3)
S[2*Np:3*Np]= S0*(p[:Np]*p[Np:2*Np])
S[3*Np:4*Np]= S0*(p[:Np]*p[2*Np:3*Np])
S[4*Np:5*Np]= S0*(p[Np:2*Np]*p[2*Np:3*Np])

#-------------------------------------------------------------------------------
# Instantiate the pystokes class
#-------------------------------------------------------------------------------
rm = pystokes.unbounded.Rbm(a, Np, 1.0/6)   # instantiate the classes
#-------------------------------------------------------------------------------
# Instantiate the pyforces class
#-------------------------------------------------------------------------------
ff = pyforces.forceFields.Forces(Np)
#-------------------------------------------------------------------------------
# Time-integration to solve the Kinematic ODEs
#-------------------------------------------------------------------------------
dt = 0.01
TimeSteps = 100
r1_array = np.zeros(TimeSteps)
r2_array = np.zeros(TimeSteps)

fil = activeFilament(dim = 3, Np = 10, b0 = 1)

fil.plotFilament()

fil.getBondAngles()

#fig = plt.figure()
#for tt in range(TimeSteps):
#    F = F*0
#    v = v*0
#    F_sed = F_sed*0
#    F_conn = F_conn*0
#    
#    # Add the Lennard-Jones potential for self-avoidance
#    ff.sedimentation(F_sed, g = -50)            # call the Sedimentation module of pyforces
##    ff.connectivity(F_conn, r, kappa, b0)
#    F_conn = connectivity(r, kappa, b0)
##    print(F_conn)
#    F = F_conn + F_sed
##    print('kappa: {}, b0: {}'.format(kappa, b0))
#
##    F[5]=0
##    ff.lennardJones(F, r, ljeps = 10)
#    
#    
#    
#    rm.stokesletV(v, r, F)
#    
#    
##    rm.stressletV(v, r, S, Nb, Nm)           # and StokesletV module of pystokes
##    rm.stressletO(omega, r, S, Nb, Nm)           # and StokesletV module of pystokes
#    
#    #    rm.stokesletV(v, r, F, Nb, Nm)
##    rm.potDipoleV(v, r, 10*p, Nb, Nm)
##    rm.potDipoleO(omega, r, 10*p, Nb, Nm)
#    
##    rm.stokesletO(omega, r, F, Nb, Nm)
#    
#    
#    r = (r + v*dt)%L
#    
#    r1_array[tt] = r[4]
#    r2_array[tt] = r[5]
#    
#    # Impose boundary conditions:
#    r[0], r[0+Np], r[0+2*Np] = L/2, L/2, L/2
##    p = (p + omega*dt)
#    
##    print(r[0])
##    print(r[1])
#    plt.cla()
#    plt.scatter(r[0], r[4], 10, marker = 'o', color ='r', alpha = 0.75)
#    plt.scatter(r[1], r[5], 10, marker = 'o', color = 'b', alpha = 0.75)
##    plt.quiver(r[0], r[4], p[0], p[4], color = 'k')
##    plt.quiver(r[1], r[5], p[1], p[5], color = 'k')
#
#    plt.xlim([0, L])
#    plt.ylim([0, L])
#    #plt.savefig('Time= %04d.png'%(tt))   # if u want to save the plots instead
#    print tt
#    plt.pause(0.001)
#plt.show()
#
#
#T = np.array(range(TimeSteps))*dt
## Plot the results
#
#plt.figure()
#
#plt.plot(T, r1_array-r2_array, 'o',color = 'k')
#plt.ylabel('Particle separation')
#plt.xlabel('Time')
#plt.show()



