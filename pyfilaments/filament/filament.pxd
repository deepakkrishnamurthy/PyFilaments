cimport cython
import numpy as np
from libc.math cimport sqrt
from cython.parallel import prange
cdef double PI = 3.14159265359

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class filament_operations:
	cdef int Np
	cdef int dim
	cdef double a 
	cdef double k
	cdef double b0
	cdef double[:] kappa_array
	cdef double [:] unit_vector_x_view
	cdef double [:] kappa_array_view
	
	cpdef get_bond_angles(self, double[:, :] dr_hat, double [:] cosAngle)
	cpdef bending_forces(self, double [:] dr, double [:, :] dr_hat, double [:] cosAngle, double[:,:] F_bending)
	cpdef connection_forces(self, double [:] dr, double [:,:] dr_hat, double [:] F_conn)




    