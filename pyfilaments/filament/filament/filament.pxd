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
	cdef double radius
	cdef double k_sc
	cdef double[:] kappa_array
	cdef double [:] kappa_hat_array_view
	cdef double [:] unit_vector
	cdef double ljrmin
	cdef double ljeps
	
	cpdef get_bond_angles(self, double[:, :] dr_hat, double [:] cosAngle)
	cpdef bending_forces(self, double [:] dr, double [:, :] dr_hat, double [:] cosAngle, double[:,:] F_bending)
	cpdef connection_forces(self, double [:] dr, double [:,:] dr_hat, double [:] F_conn)
	cpdef self_contact_forces(self, double [:] r, double [:] dr, double [:,:] dr_hat, double [:] F_sc)

	cpdef parallel_separation(self, double dri_dot_drj, double l_i, double l_j, double h_i, double h_j)
	cpdef parallel_separation_parallel_lines(self, double dri_dot_r, double l_i, double l_j)