import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
import os
from pyfilaments.activeFilaments import activeFilament
import pyfilaments.analysisutils as analysis
import imp

a = 1
b0 = 2.1*a
Np = 32
mu = 1/6.0
k = 20 # Spring constant
d0 = 1.5 # Activity strength (potential dipole)

# Create an active filament object with above parameters
fil = activeFilament(dim = 3, Np = Np, radius = a, b0 = b0, k = k, S0 = 0, D0 = d0)

filament_analysis = analysis.analysisTools(filament = fil)