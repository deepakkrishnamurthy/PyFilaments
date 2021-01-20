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
	
	def __init__(self, Np, dim, b0, k, kappa_array): 
		self.b0 = b0 
		self.Np = Np
		self.dim = dim
		self.k = k
		self.kappa_array = kappa_array
		self.unit_vector_x_view = np.array([1,0,0], dtype = np.double)
		self.kappa_array_view = np.array(self.kappa_array, dtype = np.double)

	# Calculate bond-angle vector for the filament
	cpdef get_bond_angles(self, double[:, :] dr_hat, double [:] cosAngle):
		
		cdef int ii, jj, Np = self.Np, dim = self.dim

		for ii in range(Np):
			if(ii==0):
				cosAngle[ii] = -100  # Dummy value for end points
			elif(ii == Np-1):
				cosAngle[ii] = -100 # Dummy value for end points
			else:
				cosAngle[ii] = -100
				for jj in range(dim):
					cosAngle[ii] += dr_hat[jj,ii-1]*dr_hat[jj,ii]
		
	cpdef bending_forces(self, double [:] dr, double [:, :] dr_hat, double [:] cosAngle, double[:,:] F_bending):
		# See notes for derivation of these expressions for bending potentials.
		cdef int Np = self.Np, ii, xx = 2*Np
		cdef double term_1_x, term_1_y, term_1_z, term_2_x, term_2_y, term_2_z, term_3_x, term_3_y, term_3_z
		cdef double	prefactor_1, prefactor_2_1, prefactor_2_2, prefactor_3

		for ii in range(Np):
			term_1_x, term_1_y, term_1_z = 0,0,0
			term_2_x, term_2_y, term_2_z = 0,0,0
			term_3_x, term_3_y, term_3_z = 0,0,0
			prefactor_1, prefactor_2_1, prefactor_2_2, prefactor_3 = 0,0,0,0

			# End points
			if(ii==0):
				prefactor_3 = self.kappa_array_view[ii+1]/dr[ii]

				term_3_x = prefactor_3*(dr_hat[0, ii]*cosAngle[ii+1] - dr_hat[0, ii+1])
				term_3_y = prefactor_3*(dr_hat[1, ii]*cosAngle[ii+1] - dr_hat[1, ii+1])
				term_3_z = prefactor_3*(dr_hat[2, ii]*cosAngle[ii+1] - dr_hat[2, ii+1])

			elif(ii==1):

				prefactor_2_1 = self.kappa_array_view[ii]*(1/dr[ii-1] + cosAngle[ii]/dr[ii])
				prefactor_2_2 = self.kappa_array_view[ii]*(1/dr[ii] + cosAngle[ii]/dr[ii-1])
				prefactor_3 = self.kappa_array_view[ii+1]/dr[ii]

				term_2_x = prefactor_2_1*dr_hat[0, ii] - prefactor_2_2*dr_hat[0, ii-1]
				term_2_y = prefactor_2_1*dr_hat[1, ii] - prefactor_2_2*dr_hat[1, ii-1]
				term_2_z = prefactor_2_1*dr_hat[2, ii] - prefactor_2_2*dr_hat[2, ii-1] 

				term_3_x = prefactor_3*(dr_hat[0, ii]*cosAngle[ii+1] - dr_hat[0, ii+1])
				term_3_y = prefactor_3*(dr_hat[1, ii]*cosAngle[ii+1] - dr_hat[1, ii+1])
				term_3_z = prefactor_3*(dr_hat[2, ii]*cosAngle[ii+1] - dr_hat[2, ii+1])

			elif(ii==self.Np-2):

				prefactor_1 = self.kappa_array_view[ii-1]/(dr[ii-1])
				prefactor_2_1 = self.kappa_array_view[ii]*(1/dr[ii-1] + cosAngle[ii]/dr[ii])
				prefactor_2_2 = self.kappa_array_view[ii]*(1/dr[ii] + cosAngle[ii]/dr[ii-1])

				term_1_x = prefactor_1*(dr_hat[0, ii-2] - dr_hat[0, ii-1]*cosAngle[ii-1])
				term_1_y = prefactor_1*(dr_hat[1, ii-2] - dr_hat[1, ii-1]*cosAngle[ii-1])
				term_1_z = prefactor_1*(dr_hat[2, ii-2] - dr_hat[2, ii-1]*cosAngle[ii-1])

				term_2_x = prefactor_2_1*dr_hat[0, ii] - prefactor_2_2*dr_hat[0, ii-1]
				term_2_y = prefactor_2_1*dr_hat[1, ii] - prefactor_2_2*dr_hat[1, ii-1]
				term_2_z = prefactor_2_1*dr_hat[2, ii] - prefactor_2_2*dr_hat[2, ii-1] 
			elif(ii==self.Np-1):
				prefactor_1 = self.kappa_array_view[ii-1]/(dr[ii-1])

				term_1_x = prefactor_1*(dr_hat[0, ii-2] - dr_hat[0, ii-1]*cosAngle[ii-1])
				term_1_y = prefactor_1*(dr_hat[1, ii-2] - dr_hat[1, ii-1]*cosAngle[ii-1])
				term_1_z = prefactor_1*(dr_hat[2, ii-2] - dr_hat[2, ii-1]*cosAngle[ii-1])
			else:
				# Non-endpoints 
				prefactor_1 = self.kappa_array_view[ii-1]/(dr[ii-1])
				prefactor_2_1 = self.kappa_array_view[ii]*(1/dr[ii-1] + cosAngle[ii]/dr[ii])
				prefactor_2_2 = self.kappa_array_view[ii]*(1/dr[ii] + cosAngle[ii]/dr[ii-1])
				prefactor_3 = self.kappa_array_view[ii+1]/dr[ii]

				term_1_x = prefactor_1*(dr_hat[0, ii-2] - dr_hat[0, ii-1]*cosAngle[ii-1])
				term_1_y = prefactor_1*(dr_hat[1, ii-2] - dr_hat[1, ii-1]*cosAngle[ii-1])
				term_1_z = prefactor_1*(dr_hat[2, ii-2] - dr_hat[2, ii-1]*cosAngle[ii-1])

				term_2_x = prefactor_2_1*dr_hat[0, ii] - prefactor_2_2*dr_hat[0, ii-1]
				term_2_y = prefactor_2_1*dr_hat[1, ii] - prefactor_2_2*dr_hat[1, ii-1]
				term_2_z = prefactor_2_1*dr_hat[2, ii] - prefactor_2_2*dr_hat[2, ii-1] 

				term_3_x = prefactor_3*(dr_hat[0, ii]*cosAngle[ii+1] - dr_hat[0, ii+1])
				term_3_y = prefactor_3*(dr_hat[1, ii]*cosAngle[ii+1] - dr_hat[1, ii+1])
				term_3_z = prefactor_3*(dr_hat[2, ii]*cosAngle[ii+1] - dr_hat[2, ii+1])

			F_bending[0, ii] = term_1_x + term_2_x + term_3_x
			F_bending[1, ii] = term_1_y + term_2_y + term_3_y
			F_bending[2, ii] = term_1_z + term_2_z + term_3_z
	 
	cpdef connection_forces(self, double [:] dr, double [:,:] dr_hat, double [:] F_conn):
	
		cdef int Np = self.Np, i, j, xx = 2*Np
		cdef double fx_1, fy_1, fz_1, fx_2, fy_2, fz_2, fac_1, fac_2, k = self.k, b0 = self.b0
		
		for i in range(Np):
			fx_1 = 0.0; fy_1 = 0.0; fz_1 = 0.0; fx_2 = 0.0; fy_2 = 0.0; fz_2 = 0.0;

			if(i==0):
				fac_1 = k*(dr[i] - b0)
				fx_1 = fac_1*dr_hat[0, i]
				fy_1 = fac_1*dr_hat[1, i]
				fz_1 = fac_1*dr_hat[2, i]

			elif(i==Np-1):
				fac_2 = k*(dr[i-1] - b0)
				fx_2 = fac_2*dr_hat[0, i-1]
				fy_2 = fac_2*dr_hat[1, i-1]
				fz_2 = fac_2*dr_hat[2, i-1]
			else:
				fac_1 = k*(dr[i] - b0)
				fx_1 = fac_1*dr_hat[0, i]
				fy_1 = fac_1*dr_hat[1, i]
				fz_1 = fac_1*dr_hat[2, i]

				fac_2 = k*(dr[i-1] - b0)
				fx_2 = fac_2*dr_hat[0, i-1]
				fy_2 = fac_2*dr_hat[1, i-1]
				fz_2 = fac_2*dr_hat[2, i-1]

			F_conn[i] = fx_1 - fx_2
			F_conn[i+Np] = fy_1 - fy_2
			F_conn[i+xx] = fz_1 - fz_2