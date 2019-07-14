#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:35:55 2019
Simplest Elastohydrodynamic system with hydrodynamic interactions
Two colloids, one active, one passive with one colloid clamped and an elastic spring between them.
Nearfield reupulsion using a Lennard-Jones-based potential
@author: deepak
"""
from __future__ import division
import pystokes
import pyforces
import matplotlib.pyplot as plt 
import numpy as np

#def cross(v1, v2, Np = 1, dim = 3):
#    v1, v2 = np.array(v1), np.array(v2)
#    
#    V1, V2 = np.zeros((dim, Np)),np.zeros((dim, Np))
#    
#    for ii in range(Np):
#        for jj in range(dim):
#            V1[jj, ii] = v1[ii + jj]
#            V2[jj, ii] = v2[ii + jj]
    

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


Nb, Nm = 1, 4

# Connective spring stiffness
kappa = float(5)
# Equlibrium bond-length
b0 = float(3.5*a)

# Initialize the particle positions and orientations
r[5] = r[5]-4*a
#r[1] = r[1] + 4*a

p[2*Np:3*Np] = 1

#p[0] = 1
#p[1] = -1

S0 = 5
S[:Np]      = S0*(p[:Np]*p[:Np] - 1./3)
S[Np:2*Np]  = S0*(p[Np:2*Np]*p[Np:2*Np] - 1./3)
S[2*Np:3*Np]= S0*(p[:Np]*p[Np:2*Np])
S[3*Np:4*Np]= S0*(p[:Np]*p[2*Np:3*Np])
S[4*Np:5*Np]= S0*(p[Np:2*Np]*p[2*Np:3*Np])

dt = 0.05
# Instantiate the pystokes class
rm = pystokes.unbounded.Rbm(a, Np, 1.0/6)   # instantiate the classes
# Instantiate the pyforces class
ff = pyforces.forceFields.Forces(Np)

TimeSteps = 200
r1_array = np.zeros(TimeSteps)
r2_array = np.zeros(TimeSteps)

fig = plt.figure()
for tt in range(TimeSteps):
    F = F*0
    v = v*0
    F_sed = F_sed*0
    F_conn = F_conn*0
    
    # Add the Lennard-Jones potential for self-avoidance
#    ff.sedimentation(F_sed, g = -100)            # call the Sedimentation module of pyforces
#    ff.connectivity(F_conn, r, kappa, b0)
    F_conn = connectivity(r, kappa, b0)
#    print(F_conn)
    F = F_conn + F_sed
#    print('kappa: {}, b0: {}'.format(kappa, b0))

#    F[5]=0
#    ff.lennardJones(F, r, ljeps = 10)
    
    
    
    rm.stokesletV(v, r, F)
    
    
#    rm.stressletV(v, r, S, Nb, Nm)           # and StokesletV module of pystokes
#    rm.stressletO(omega, r, S, Nb, Nm)           # and StokesletV module of pystokes
    
    #    rm.stokesletV(v, r, F, Nb, Nm)
#    rm.potDipoleV(v, r, 10*p, Nb, Nm)
#    rm.potDipoleO(omega, r, 10*p, Nb, Nm)
    
#    rm.stokesletO(omega, r, F, Nb, Nm)
    
    
    r = (r + v*dt)%L
    
    r1_array[tt] = r[4]
    r2_array[tt] = r[5]
    
    # Impose boundary conditions:
    r[0], r[0+Np], r[0+2*Np] = L/2, L/2, L/2
#    p = (p + omega*dt)
    
#    print(r[0])
#    print(r[1])
    plt.cla()
    plt.scatter(r[0], r[4], 10, marker = 'o', color ='r', alpha = 0.75)
    plt.scatter(r[1], r[5], 10, marker = 'o', color = 'b', alpha = 0.75)
#    plt.quiver(r[0], r[4], p[0], p[4], color = 'k')
#    plt.quiver(r[1], r[5], p[1], p[5], color = 'k')

    plt.xlim([0, L])
    plt.ylim([0, L])
    #plt.savefig('Time= %04d.png'%(tt))   # if u want to save the plots instead
    print tt
    plt.pause(0.001)
plt.show()


T = np.array(range(TimeSteps))*dt
# Plot the results

plt.figure()

plt.plot(T, r1_array-r2_array, 'o',color = 'k')
plt.ylabel('Particle separation')
plt.xlabel('Time')
plt.show()



