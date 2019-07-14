#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:05:25 2019
Rigib-body-motion of hydrodynamically interacting particles
@author: deepak
"""

import pystokes
import pyforces
import numpy as np
eta = 1.0/6
a, Np, dim = 1, 3, 3                 # radius and number of particles and dimension
v = np.zeros(dim*Np)                 # Memory allocation for velocity
r = np.zeros(dim*Np)                 # Position vector of the particles
F = np.zeros(dim*Np)                 # Forces on the particles

r[0], r[1], r[2] = -2, 0 , 2         # x-comp of PV of particles

# instantiate the pystokes classes with Rigid-Body-Motion
pRbm = pystokes.unbounded.Rbm(a, Np, eta)    
# Instantiate the pyforces class
ff = pyforces.forceFields.Forces(Np)

print 'Initial velocity', v

ff.sedimentation(F, g=10)            # call the Sedimentation module of pyforces
pRbm.stokesletV(v, r, F)               # and StokesletV module of pystokes

pRbm.potDipoleV()


print 'Updated velocity', v