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

		# Temporary variables


	cdef double [:] vector_addition(self, double [:] vector_1, double [:] vector_2):

		cdef int ii
		result = np.zeros(len(vector_1))
		cdef double [:] result_view = result

		for ii in range(len(vector_1)):
			result_view[ii] = vector_1[ii] + vector_2[ii]

		return result_view

	# Calculate bond-angle vector for the filament
	cpdef get_bond_angles(self, double[:, :] dr_hat, double [:] cosAngle):
		
		cdef int ii, jj, Np = self.Np, dim = self.dim

		unit_vector_x = np.array([1,0,0], dtype = np.double)

		cdef double [:] unit_vector_x_view = unit_vector_x
		
		
		for ii in prange(Np, nogil=True):

			if(ii==0 or ii == Np-1):
				cosAngle[ii] = 0
				for jj in range(dim):
					cosAngle[ii] += dr_hat[jj,ii]*unit_vector_x_view[jj]
			else:
				cosAngle[ii] = 0
				for jj in range(dim):
					cosAngle[ii] += dr_hat[jj,ii-1]*dr_hat[jj,ii]
		return

		
	# Find the local tangent vector of the filament at the position of each particle
	cpdef get_tangent_vectors(self, double[:, :] dr_hat, double [:, :] t_hat):
		
		cdef int ii, jj, Np = self.Np, dim = self.dim
		
		vector = np.zeros(self.dim, dtype = np.double)

		cdef double[:] vector_view = vector

		cdef double[:] vector_mag = np.zeros(self.Np, dtype = np.double)
		cdef double [:] vector_mag_2 = np.zeros(self.Np, dtype = np.double)

		# @@@ Not parallelizing these for now. 
		for ii in prange(Np, nogil = True):
			
			if ii==0:
				t_hat[:,ii] = dr_hat[:,ii]
			elif ii == Np-1:
				t_hat[:,-1] = dr_hat[:,-1]
			else:
				# For particles that have a neighbhor on each side, the tangent vector is the average of the two bonds. 
				vector_mag_2[ii] = 0

				for jj in range(dim):
					vector_view[jj] = dr_hat[jj,ii-1] + dr_hat[jj,ii]
					vector_mag_2[ii] += vector_view[jj]**2
				
				vector_mag[ii] = sqrt(vector_mag_2[ii])
				for jj in range(dim):
					t_hat[jj,ii] = (vector_view[jj])/(vector_mag[ii])
				# t_hat[:,ii] = (vector_view)*(1/vector_mag)
		return
				

	cpdef bending_forces(self, double [:] dr, double [:, :] dr_hat, double [:] cosAngle, double[:,:] F_bending):
		# See notes for derivation of these expressions for bending potentials.
		cdef int Np = self.Np, ii, jj, xx = 2*Np, dim = self.dim
		cdef double [:] kappa_array_view = np.array(self.kappa_array, dtype = np.double)
		cdef double [:] term_n_minus = np.zeros(self.dim, dtype = np.double)
		cdef double [:] term_n = np.zeros(self.dim, dtype = np.double)
		cdef double [:] term_n_plus = np.zeros(self.dim, dtype = np.double)
		cdef double [:] term_n1 = np.zeros(self.dim, dtype = np.double)
		cdef double [:] term_n2 = np.zeros(self.dim, dtype = np.double)
		cdef double [:] temp_vect = np.zeros(self.dim, dtype = np.double)

		# @@@ Not parallelizing these for now. 
		for ii in prange(Np, nogil=True):
			

			if ii==0:
				# Torque-free ends
				# F_bending[:,ii] = kappa_array_view[ii+1]*(1/dr[ii])*(dr_hat[:, ii]*cosAngle[ii+1] - dr_hat[:, ii+1])
				for jj in range(dim):
					temp_vect[jj] = dr_hat[jj, ii]*cosAngle[ii+1]  - dr_hat[jj, ii+1]
					F_bending[jj,ii] = kappa_array_view[ii+1]*(1/dr[ii])*(temp_vect[jj])

			elif ii == Np-1:
				# Torque-free ends
				for jj in range(dim):
					temp_vect[jj] = dr_hat[jj, ii-2] - cosAngle[ii - 1]*dr_hat[jj, ii-1]
					F_bending[jj,ii] = kappa_array_view[ii-1]*(1/dr[ii-1])*(temp_vect[jj])
		
			else:
				
				if(ii!=1):
					for jj in range(dim):
						temp_vect[jj] = dr_hat[jj, ii-2] - dr_hat[jj, ii-1]*cosAngle[ii-1]
						term_n_minus[jj] = kappa_array_view[ii-1]*(temp_vect[jj])*(1/dr[ii-1])
					
				elif(ii != Np-2):
					for jj in range(dim):
						term_n_plus[jj] = kappa_array_view[ii+1]*(-dr_hat[jj, ii+1] + dr_hat[jj, ii]*cosAngle[ii + 1])*(1/dr[ii])
				else:
					for jj in range(dim):
						term_n_minus[jj] = 0
						term_n_plus[jj] = 0

				for jj in range(dim):
					term_n1[jj] = (1/dr[ii-1] + cosAngle[ii]/dr[ii])*dr_hat[jj, ii]
					term_n2[jj] = -(1/dr[ii] + cosAngle[ii]/dr[ii-1])*dr_hat[jj, ii-1]
				
					term_n[jj] = kappa_array_view[ii]*(term_n1[jj] + term_n2[jj])
				
					
					
					F_bending[jj,ii] = term_n_minus[jj] + term_n[jj] + term_n_plus[jj]
				
		return
		# # Now reshape the forces array
		# self.F_bending_array = self.reshapeToArray(self.F_bending)  
 
	
	cpdef connection_forces(self, double [:] r, double [:] F_conn):
	
		cdef int Np = self.Np, i, j, xx = 2*Np
		cdef double dx, dy, dz, dr2, dr, idr, fx, fy, fz, fac, k = self.k, b0 = self.b0
		

		for i in prange(Np, nogil=True):
			fx = 0.0; fy = 0.0; fz = 0.0;

			for j in range(i, Np):
				
				if((i-j)==1 or (i-j)==-1):
					
					dx = r[i   ] - r[j   ]
					dy = r[i+Np] - r[j+Np]
					dz = r[i+xx] - r[j+xx] 
					dr2 = dx*dx + dy*dy + dz*dz
					dr = sqrt(dr2)
					
	#                    dr_hat = np.array([dx, dy, dz], dtype = 'float')*(1/dr)
					
					fac = -k*(dr - b0)
				
					fx = fac*dx/dr
					fy = fac*dy/dr
					fz = fac*dz/dr
					
					
					# Force on particle "i"
					F_conn[i]    += fx 
					F_conn[i+self.Np] += fy 
					F_conn[i+xx] += fz 
					
					# Force on particle "j"
					F_conn[j]    -= fx 
					F_conn[j+self.Np] -= fy 
					F_conn[j+xx] -= fz 

		return
