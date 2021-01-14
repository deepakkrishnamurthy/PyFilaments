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
		self.vector_view = np.zeros(self.dim, dtype = np.double)

		self.vector_mag = np.zeros(self.Np, dtype = np.double)
		self.vector_mag_2 = np.zeros(self.Np, dtype = np.double)

		self.kappa_array_view = np.array(self.kappa_array, dtype = np.double)
		self.term_n_minus = np.zeros(self.dim, dtype = np.double)
		self.term_n = np.zeros(self.dim, dtype = np.double)
		self.term_n_plus = np.zeros(self.dim, dtype = np.double)
		self.term_n1 = np.zeros(self.dim, dtype = np.double)
		self.term_n2 = np.zeros(self.dim, dtype = np.double)
		self.temp_vect = np.zeros(self.dim, dtype = np.double)
		# Temporary variables
		pass

	# Calculate bond-angle vector for the filament
	cpdef get_bond_angles(self, double[:, :] dr_hat, double [:] cosAngle):
		
		cdef int ii, jj, Np = self.Np, dim = self.dim

		for ii in prange(Np, nogil = True):
			cosAngle[ii] = 0
			if(ii==0):
				cosAngle[ii] = 0
				for jj in range(dim):
					cosAngle[ii] += dr_hat[jj,ii]*self.unit_vector_x_view[jj]
			elif(ii == Np-1):
				cosAngle[ii] = 0
				for jj in range(dim):
					cosAngle[ii] += dr_hat[jj,ii-1]*self.unit_vector_x_view[jj]
			else:
				cosAngle[ii] = 0
				for jj in range(dim):
					cosAngle[ii] += dr_hat[jj,ii-1]*dr_hat[jj,ii]

		

		
	# # Find the local tangent vector of the filament at the position of each particle
	cpdef get_tangent_vectors(self, double[:, :] dr_hat, double [:, :] t_hat):
		
		cdef int ii, jj, Np = self.Np, dim = self.dim
		
		# vector = np.zeros(self.dim, dtype = np.double)

		# cdef double[:] vector_view = vector

		# cdef double[:] vector_mag = np.zeros(self.Np, dtype = np.double)
		# cdef double [:] vector_mag_2 = np.zeros(self.Np, dtype = np.double)

		t_hat[:,0] = dr_hat[:,0]
		t_hat[:,-1] = dr_hat[:,-1]
		# @@@ Not parallelizing these for now. 
		for ii in prange(1, Np-1, nogil = True):
			# For particles that have a neighbhor on each side, the tangent vector is the average of the two bonds. 
			self.vector_mag_2[ii] = 0

			for jj in range(dim):
				self.vector_view[jj] = dr_hat[jj,ii-1] + dr_hat[jj,ii]
				self.vector_mag_2[ii] += self.vector_view[jj]**2
			
			self.vector_mag[ii] = sqrt(self.vector_mag_2[ii])
			for jj in range(dim):
				t_hat[jj,ii] = (self.vector_view[jj])/(self.vector_mag[ii])
			# t_hat[:,ii] = (vector_view)*(1/vector_mag)
		# return
				

	cpdef bending_forces(self, double [:] dr, double [:, :] dr_hat, double [:] cosAngle, double[:,:] F_bending):
		# See notes for derivation of these expressions for bending potentials.
		cdef int Np = self.Np, ii, jj, xx = 2*Np, dim = self.dim
		# cdef double [:] kappa_array_view = np.array(self.kappa_array, dtype = np.double)
		# cdef double [:] term_n_minus = np.zeros(self.dim, dtype = np.double)
		# cdef double [:] term_n = np.zeros(self.dim, dtype = np.double)
		# cdef double [:] term_n_plus = np.zeros(self.dim, dtype = np.double)
		# cdef double [:] term_n1 = np.zeros(self.dim, dtype = np.double)
		# cdef double [:] term_n2 = np.zeros(self.dim, dtype = np.double)
		# cdef double [:] temp_vect = np.zeros(self.dim, dtype = np.double)

		# @@@ Not parallelizing these for now. 
		for ii in prange(Np, nogil=True):
			if ii==0:
				# Torque-free ends
				# F_bending[:,ii] = kappa_array_view[ii+1]*(1/dr[ii])*(dr_hat[:, ii]*cosAngle[ii+1] - dr_hat[:, ii+1])
				for jj in range(dim):
					self.temp_vect[jj] = dr_hat[jj, ii]*cosAngle[ii+1]  - dr_hat[jj, ii+1]
					F_bending[jj,ii] = self.kappa_array_view[ii+1]*(1/dr[ii])*(self.temp_vect[jj])

			elif ii == Np-1:
				# Torque-free ends
				for jj in range(dim):
					self.temp_vect[jj] = dr_hat[jj, ii-2] - cosAngle[ii - 1]*dr_hat[jj, ii-1]
					F_bending[jj,ii] = self.kappa_array_view[ii-1]*(1/dr[ii-1])*(self.temp_vect[jj])
		
			else:
				
				if(ii!=1):
					for jj in range(dim):
						self.temp_vect[jj] = dr_hat[jj, ii-2] - dr_hat[jj, ii-1]*cosAngle[ii-1]
						self.term_n_minus[jj] = self.kappa_array_view[ii-1]*(self.temp_vect[jj])*(1/dr[ii-1])
				else:
					for jj in range(dim):
						self.term_n_minus[jj] = 0

				if(ii != Np-2):
					for jj in range(dim):
						self.term_n_plus[jj] = self.kappa_array_view[ii+1]*(-dr_hat[jj, ii+1] + dr_hat[jj, ii]*cosAngle[ii + 1])*(1/dr[ii])
				else:
					for jj in range(dim):
						self.term_n_plus[jj] = 0

				for jj in range(dim):
					self.term_n1[jj] = (1/dr[ii-1] + cosAngle[ii]/dr[ii])*dr_hat[jj, ii]
					self.term_n2[jj] = -(1/dr[ii] + cosAngle[ii]/dr[ii-1])*dr_hat[jj, ii-1]
				
					self.term_n[jj] = self.kappa_array_view[ii]*(self.term_n1[jj] + self.term_n2[jj])
					F_bending[jj,ii] = self.term_n_minus[jj] + self.term_n[jj] + self.term_n_plus[jj]
				
		# return
	# 	# # Now reshape the forces array
	# 	# self.F_bending_array = self.reshapeToArray(self.F_bending)  
 
	
	cpdef connection_forces(self, double [:] dr, double [:,:] dr_hat, double [:] F_conn):
	
		cdef int Np = self.Np, i, j, xx = 2*Np
		cdef double fx_1, fy_1, fz_1, fx_2, fy_2, fz_2, fac_1, fac_2, k = self.k, b0 = self.b0
		
		for i in prange(Np, nogil = True):
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