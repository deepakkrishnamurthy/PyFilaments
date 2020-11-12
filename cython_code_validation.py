from pyfilaments.activeFilaments import activeFilament
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt 
from sys import platform


bc = {0:'free', -1:'free'}

fil = activeFilament(dim = 3, Np = 32, radius = 1, b0 = 4, k = 10, S0 = 0, D0 = 0, bc = bc)

# fil.plotFilament(r = fil.r0)

fil.getSeparationVector()


print(fil.dr_hat)



# Testing the Bond Angles function
fil.filament.get_bond_angles(fil.dr_hat, fil.cosAngle)



print('Joint angles')
print(fil.cosAngle)

cython_angles = fil.cosAngle


fil.getBondAngles()

plt.figure()
plt.plot(fil.cosAngle, 'gs')
plt.plot(cython_angles, 'ro', alpha = 0.5)
plt.title('Bond angles')
plt.show()
print('Error in bending angles computation: {}'.format(np.sum(fil.cosAngle - cython_angles)))


# Testing the connection forces function
fil.filament.connection_forces(fil.r, fil.F_conn)

F_conn_cython = fil.F_conn

fil.ConnectionForces()

plt.figure()
plt.plot(fil.F_conn, 'gs')
plt.plot(F_conn_cython, 'ro', alpha = 0.5)
plt.title('Connection Forces')
plt.show()

print('Error in connection forces computation: {}'.format(np.sum(fil.F_conn - F_conn_cython)))


# Testing the Tangent vectors function

fil.filament.get_tangent_vectors(fil.dr_hat, fil.t_hat)

tangent_vectors_cython = fil.t_hat

fil.getTangentVectors()

plt.figure()
plt.plot(fil.t_hat, 'gs')
plt.plot(tangent_vectors_cython, 'ro', alpha = 0.5)
plt.title('Tangent vectors')
plt.show()

print('Error in tangent vector computation: {}'.format(np.sum(fil.t_hat - tangent_vectors_cython)))


# Testing bending forces function
fil.filament.bending_forces(fil.dr, fil.dr_hat, fil.cosAngle, fil.F_bending)

F_bending_cython = fil.F_bending

fil.BendingForces()

plt.figure()
plt.plot(fil.t_hat, 'gs')
plt.plot(tangent_vectors_cython, 'ro', alpha = 0.5)
plt.title('Bending forces')
plt.show()
print('Error in bending forces computation: {}'.format(np.sum(fil.F_bending - F_bending_cython)))




