cimport cython
import numpy as np
from libc.math cimport sqrt, pow
from cython.parallel import prange
cdef double PI = 3.14159265359

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class filament_operations:
	
	def __init__(self, Np, dim, radius, b0, k, kappa_hat_array, unit_vector = [1,0,0], ljrmin = 3, ljeps = 0.01): 
		self.radius = radius
		self.b0 = b0 
		self.Np = Np
		self.dim = dim
		self.k = k
		self.kappa_hat_array = kappa_hat_array
		self.k_sc = 1000
		self.unit_vector = np.array(unit_vector, dtype = np.double)
		self.kappa_hat_array_view = np.array(self.kappa_hat_array, dtype = np.double)
		self.ljrmin = ljrmin
		self.ljeps = ljeps

	# Calculate bond-angle vector for the filament
	cpdef get_bond_angles(self, double[:, :] dr_hat, double [:] cosAngle):
		
		cdef int ii, jj, Np = self.Np, dim = self.dim
		cdef double unit_vector_x = self.unit_vector[0], unit_vector_y = self.unit_vector[1], unit_vector_z = self.unit_vector[2] 

		for ii in range(Np):
			if(ii==0):
				cosAngle[ii] = unit_vector_x*dr_hat[0,ii] + unit_vector_y*dr_hat[1,ii] + unit_vector_z*dr_hat[2,ii]  # Clamped end
			elif(ii == Np-1):
				# cosAngle[ii] = dr_hat[0,ii-1]*unit_vector_x + dr_hat[1,ii-1]*unit_vector_y + dr_hat[2,ii-1]*unit_vector_z 	# Dummy value for free-end
				cosAngle[ii] = 1.0
			else:
				cosAngle[ii] = 0
				for jj in range(dim):
					cosAngle[ii] += dr_hat[jj,ii-1]*dr_hat[jj,ii]
		
	cpdef bending_forces(self, double [:] dr, double [:, :] dr_hat, double [:] cosAngle, double[:,:] F_bending):
		# See notes for derivation of these expressions for bending potentials.
		cdef int Np = self.Np, ii, xx = 2*Np
		cdef double term_1_x, term_1_y, term_1_z, term_2_x, term_2_y, term_2_z, term_3_x, term_3_y, term_3_z
		cdef double	prefactor_1, prefactor_2_1, prefactor_2_2, prefactor_3
		cdef double b0 = self.b0
		cdef double unit_vector_x = self.unit_vector[0], unit_vector_y = self.unit_vector[1], unit_vector_z = self.unit_vector[2] 


		for ii in range(Np):
			term_1_x = 0.0; term_1_y = 0.0; term_1_z = 0.0;
			term_2_x = 0.0; term_2_y = 0.0; term_2_z = 0.0;
			term_3_x = 0.0; term_3_y = 0.0; term_3_z = 0.0;
			prefactor_1 = 0.0; prefactor_2_1 = 0.0; prefactor_2_2 = 0.0; prefactor_3 = 0.0;

			# End points
			if(ii == 0):
				# Terminal colloid:0
				prefactor_2_1 = self.kappa_hat_array_view[ii]*(1/b0 + cosAngle[ii]/dr[ii])
				prefactor_2_2 = self.kappa_hat_array_view[ii]*(1/dr[ii] + cosAngle[ii]/b0)
				prefactor_3 = self.kappa_hat_array_view[ii+1]/dr[ii]

				term_2_x = prefactor_2_1*dr_hat[0, ii] - prefactor_2_2*unit_vector_x
				term_2_y = prefactor_2_1*dr_hat[1, ii] - prefactor_2_2*unit_vector_y
				term_2_z = prefactor_2_1*dr_hat[2, ii] - prefactor_2_2*unit_vector_z

				term_3_x = prefactor_3*(dr_hat[0, ii]*cosAngle[ii+1] - dr_hat[0, ii+1])
				term_3_y = prefactor_3*(dr_hat[1, ii]*cosAngle[ii+1] - dr_hat[1, ii+1])
				term_3_z = prefactor_3*(dr_hat[2, ii]*cosAngle[ii+1] - dr_hat[2, ii+1])

			elif(ii == 1):
				# Next to terminal colloid: 1
				prefactor_1 = self.kappa_hat_array_view[ii-1]/(dr[ii-1])
				prefactor_2_1 = self.kappa_hat_array_view[ii]*(1/dr[ii-1] + cosAngle[ii]/dr[ii])
				prefactor_2_2 = self.kappa_hat_array_view[ii]*(1/dr[ii] + cosAngle[ii]/dr[ii-1])
				prefactor_3 = self.kappa_hat_array_view[ii+1]/dr[ii]

				term_1_x = prefactor_1*(unit_vector_x - dr_hat[0, ii-1]*cosAngle[ii-1])
				term_1_y = prefactor_1*(unit_vector_y - dr_hat[1, ii-1]*cosAngle[ii-1])
				term_1_z = prefactor_1*(unit_vector_z - dr_hat[2, ii-1]*cosAngle[ii-1])

				term_2_x = prefactor_2_1*dr_hat[0, ii] - prefactor_2_2*dr_hat[0, ii-1]
				term_2_y = prefactor_2_1*dr_hat[1, ii] - prefactor_2_2*dr_hat[1, ii-1]
				term_2_z = prefactor_2_1*dr_hat[2, ii] - prefactor_2_2*dr_hat[2, ii-1] 

				term_3_x = prefactor_3*(dr_hat[0, ii]*cosAngle[ii+1] - dr_hat[0, ii+1])
				term_3_y = prefactor_3*(dr_hat[1, ii]*cosAngle[ii+1] - dr_hat[1, ii+1])
				term_3_z = prefactor_3*(dr_hat[2, ii]*cosAngle[ii+1] - dr_hat[2, ii+1])

			elif(ii == Np-2):
				# Next to terminal colloid: Np-2
				prefactor_1 = self.kappa_hat_array_view[ii-1]/(dr[ii-1])
				prefactor_2_1 = self.kappa_hat_array_view[ii]*(1/dr[ii-1] + cosAngle[ii]/dr[ii])
				prefactor_2_2 = self.kappa_hat_array_view[ii]*(1/dr[ii] + cosAngle[ii]/dr[ii-1])
				prefactor_3 = self.kappa_hat_array_view[ii+1]/dr[ii]

				term_1_x = prefactor_1*(dr_hat[0, ii-2] - dr_hat[0, ii-1]*cosAngle[ii-1])
				term_1_y = prefactor_1*(dr_hat[1, ii-2] - dr_hat[1, ii-1]*cosAngle[ii-1])
				term_1_z = prefactor_1*(dr_hat[2, ii-2] - dr_hat[2, ii-1]*cosAngle[ii-1])

				term_2_x = prefactor_2_1*dr_hat[0, ii] - prefactor_2_2*dr_hat[0, ii-1]
				term_2_y = prefactor_2_1*dr_hat[1, ii] - prefactor_2_2*dr_hat[1, ii-1]
				term_2_z = prefactor_2_1*dr_hat[2, ii] - prefactor_2_2*dr_hat[2, ii-1] 

				term_3_x = prefactor_3*(dr_hat[0, ii]*cosAngle[ii+1] - unit_vector_x)
				term_3_y = prefactor_3*(dr_hat[1, ii]*cosAngle[ii+1] - unit_vector_y)
				term_3_z = prefactor_3*(dr_hat[2, ii]*cosAngle[ii+1] - unit_vector_z)

			elif(ii == Np-1):
				# Terminal colloid: Np-1

				prefactor_1 = self.kappa_hat_array_view[ii-1]/(dr[ii-1])
				prefactor_2_1 = self.kappa_hat_array_view[ii]*(1/dr[ii-1] + cosAngle[ii]/b0)
				prefactor_2_2 = self.kappa_hat_array_view[ii]*(1/b0 + cosAngle[ii]/dr[ii-1])

				term_1_x = prefactor_1*(dr_hat[0, ii-2] - dr_hat[0, ii-1]*cosAngle[ii-1])
				term_1_y = prefactor_1*(dr_hat[1, ii-2] - dr_hat[1, ii-1]*cosAngle[ii-1])
				term_1_z = prefactor_1*(dr_hat[2, ii-2] - dr_hat[2, ii-1]*cosAngle[ii-1])

				term_2_x = prefactor_2_1*unit_vector_x - prefactor_2_2*dr_hat[0, ii-1]
				term_2_y = prefactor_2_1*unit_vector_y - prefactor_2_2*dr_hat[1, ii-1]
				term_2_z = prefactor_2_1*unit_vector_z - prefactor_2_2*dr_hat[2, ii-1] 
		
			else:
				# Interior colloids
				prefactor_1 = self.kappa_hat_array_view[ii-1]/(dr[ii-1])
				prefactor_2_1 = self.kappa_hat_array_view[ii]*(1/dr[ii-1] + cosAngle[ii]/dr[ii])
				prefactor_2_2 = self.kappa_hat_array_view[ii]*(1/dr[ii] + cosAngle[ii]/dr[ii-1])
				prefactor_3 = self.kappa_hat_array_view[ii+1]/dr[ii]

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

	cpdef self_contact_forces(self, double [:] r, double [:] dr, double [:,:] dr_hat, double [:] F_sc):

		cdef int Np = self.Np, i, j, xx = 2*Np, index
		cdef double h_i, h_j, dmin_x, dmin_y, dmin_z, dmin_2, epsilon, gamma_final, delta_final, d_perp2 
		cdef double f_x, f_y, f_z, dx, dy, dz, dr_ij, dr_ij1, dr_i1j, dr_i1j1, dri_dot_drj, dri_dot_r, drj_dot_r
		cdef double b0 = self.b0, radius = self.radius, k_sc = self.k_sc
		cdef double min_distance = (b0*b0) + 4.0*radius*radius, torque_scale_factor
		cdef double ljrmin = self.ljrmin, ljeps = self.ljeps,  idr, rminbyr, fac

		for i in range(Np-1):
			f_x = 0.0; f_y = 0.0; f_z = 0.0;
			
			for j in range(Np-1):
				torque_scale_factor = 0
				h_i = 0.0; h_j = 0.0; gamma_final = 0.0; delta_final = 0.0;
 
				dx = r[i   ] - r[j+1]
				dy = r[i+Np] - r[j+1+Np]
				dz = r[i+xx] - r[j+1+xx] 
				dr_ij1 = dx*dx + dy*dy + dz*dz
			
				dx = r[i+1] - r[j   ]
				dy = r[i+1+Np] - r[j+Np]
				dz = r[i+1+xx] - r[j+xx] 
				dr_i1j = dx*dx + dy*dy + dz*dz
			
				dx = r[i+1] - r[j+1]
				dy = r[i+1+Np] - r[j+1+Np]
				dz = r[i+1+xx] - r[j+1+xx] 
				dr_i1j1 = dx*dx + dy*dy + dz*dz

				dx = r[i   ] - r[j   ]
				dy = r[i+Np] - r[j+Np]
				dz = r[i+xx] - r[j+xx] 
				dr_ij = dx*dx + dy*dy + dz*dz
			
				# if i!=j and abs(i-j)>int(Np/4) and (dr_ij < min_distance or dr_ij1 < min_distance or dr_i1j < min_distance or dr_i1j1 < min_distance):  # No need to check for interactions between very close spheres
				if i!=j and abs(i-j)> 2:

					# Check if the lines are parallel
					dri_dot_drj = 0; dri_dot_r = 0; drj_dot_r = 0;
					for index in range(3):
						dri_dot_drj += dr_hat[index, i]*dr_hat[index, j]
						dri_dot_r += dr_hat[index, i]*(r[i + index*Np] - r[j + index*Np])
						drj_dot_r += dr_hat[index, j]*(r[i + index*Np] - r[j + index*Np])
					
					if((dri_dot_drj - 1.0)*(dri_dot_drj - 1.0) < 1e-6):
						# The lines are parallel or nearly parallel
						# Find the distance between the parallel lines
						h_i = 0
						h_j = 0

						d_min2 = 0.0
						for index in range(3):
							d_min2 = (r[i + index*Np] - r[j + index*Np] - dri_dot_r*dr_hat[index, i])*(r[i + index*Np] - r[j + index*Np] - dri_dot_r*dr_hat[index, i])

						# Check this against the minimum distance between two rods
						if(d_min2 <= ljrmin*ljrmin):
							# The parallel rods are close enough to interact. Find the actual closest line and direction of the interaction force vector.
							gamma_final, delta_final = self.parallel_separation_parallel_lines(dri_dot_r, dr[i], dr[j])

						else:
							# The rods are parallel but are not close enough to interact, in thsi case the interaction force is zero.
							f_x = 0
							f_y = 0
							f_z = 0
							break
					else:
						# Non-parallel case.
					
						# Find the common normal to both rods.
						h_i = (drj_dot_r*dri_dot_drj - dri_dot_r)/(1 - dri_dot_drj*dri_dot_drj)
						h_j = (drj_dot_r - dri_dot_drj*dri_dot_r)/(1 - dri_dot_drj*dri_dot_drj)

						# Find the in-plane points of cloest-approach 
						if(h_i >= 0 and h_i <= dr[i] and h_j >= 0 and h_j <= dr[j]):
							gamma_final, delta_final = 0,0 
						else:
							gamma_final, delta_final = self.parallel_separation(dri_dot_drj, dr[i], dr[j], h_i, h_j)

					# Find the line of closest approach between the two rods based on the calculated parameters. 
					dmin_x = r[i   ] - r[j   ] + (h_i + gamma_final)*dr_hat[0, i] - (h_j + delta_final)*dr_hat[0, j]
					dmin_y = r[i+Np] - r[j+Np] + (h_i + gamma_final)*dr_hat[1, i] - (h_j + delta_final)*dr_hat[1, j]
					dmin_z = r[i+xx] - r[j+xx] + (h_i + gamma_final)*dr_hat[2, i] - (h_j + delta_final)*dr_hat[2, j]

					dmin_2	= dmin_x*dmin_x +dmin_y*dmin_y + dmin_z*dmin_z

					# epsilon = (2*radius - sqrt(dmin_2))

					if(dmin_2 <= (ljrmin*ljrmin)):
						idr     = 1.0/sqrt(dmin_2)
						rminbyr = ljrmin*idr 
						fac   = ljeps*(pow(rminbyr, 12) - pow(rminbyr, 6))*idr*idr
						# if(epsilon>0):
						f_x += fac*dmin_x
						f_y += fac*dmin_y
						f_z += fac*dmin_z
						torque_scale_factor = (h_i + gamma_final)/dr[i]

					# torque_scale_factor = (h_j + delta_final)/dr[j]
					# # Force on colloid j
					# F_sc[j] = -f_x*(1-torque_scale_factor)
					# F_sc[j+Np] = -f_y*(1-torque_scale_factor)
					# F_sc[j+xx] = -f_z*(1-torque_scale_factor)

					# # Force on colloid j+1 
					# F_sc[j + 1] = -f_x*torque_scale_factor
					# F_sc[j+1+Np] = -f_y*torque_scale_factor
					# F_sc[j+1+xx] = -f_z*torque_scale_factor

			# Force on colloid i
			
			F_sc[i] += f_x*(1 - torque_scale_factor)
			F_sc[i+Np] += f_y*(1 - torque_scale_factor)
			F_sc[i+xx] += f_z*(1 - torque_scale_factor)

			# Force on colloid i+1 
			F_sc[i + 1] += f_x*torque_scale_factor
			F_sc[i+1+Np] += f_y*torque_scale_factor
			F_sc[i+1+xx] += f_z*torque_scale_factor

			
		return

	cpdef parallel_separation(self, double dri_dot_drj, double l_i, double l_j, double h_i, double h_j):
		"""
		Adapted from Allen et al. Adv. Chem. Phys. Vol LXXXVI, 1003, p.1.

		Takes input of two line-segments and returns the shortest distance and parameters giving the shortest distance vector 
		in the plane parallel to both line segments
		Returns: lambda_min, delta_min
		"""
		cdef double gamma_1, gamma_2, delta_1, delta_2, gamma_m, delta_m, gamma_min, delta_min, gamma_final, delta_final
		cdef double a1, a2, b1, b2, f1, f2

		gamma_1 = -h_i 
		gamma_2 = -h_i + l_i
		gamma_m = gamma_1
		# Choose the line closest to the origin
		if(gamma_1*gamma_1 > gamma_2*gamma_2):
			gamma_m = gamma_2

		delta_1 = -h_j
		delta_2 = -h_j +l_j
		delta_m = delta_1

		if(delta_1*delta_1 > delta_2*delta_2):
			delta_m = delta_2

		# Optimize delta on gamma_m
		gamma_min = gamma_m
		delta_min = gamma_m*dri_dot_drj

		if(delta_min + h_j >= 0 and delta_min + h_j <= l_j):
			delta_min = delta_min
		else:
			delta_min = delta_1
			a1 = delta_min - delta_1
			a2 = delta_min - delta_2
			if(a1*a1 > a2*a2):
				delta_min = delta_2

		# Distance at this gamma and delta value
		f1 = gamma_min*gamma_min + delta_min*delta_min - 2*gamma_min*delta_min*dri_dot_drj
		gamma_final = gamma_min
		delta_final = delta_min

		# Now choose the line delta_m and optimize gamma
		delta_min = delta_m
		gamma_min = delta_m*dri_dot_drj

		if(gamma_min + h_i >= 0 and gamma_min + h_i <= l_i):
			gamma_min = gamma_min
		else:
			gamma_min = gamma_1
			b1 = gamma_min - gamma_1
			b2 = gamma_min - gamma_2
			if(b1*b1 > b2*b2):
				gamma_min = gamma_2

		f2 = gamma_min*gamma_min + delta_min*delta_min - 2*gamma_min*delta_min*dri_dot_drj
		
		if(f1 < f2):
			pass
		else:
			delta_final = delta_min
			gamma_final = gamma_min

		return gamma_final, delta_final

	cpdef parallel_separation_parallel_lines(self, double dri_dot_r, double l_i, double l_j):
		"""
			Find the line of closest separtion between points lying on parallel lines.
			Handles both cases:
			1. Overalapping lines.
			2. No overlapping lines

		"""
		cdef double gamma_min, gamma_max, delta_min, delta_max, gamma_final, delta_final

		#Find gamma_min
		if(-dri_dot_r<0):
			gamma_min = 0.0
		elif (-dri_dot_r >= 0 and -dri_dot_r < l_i):
			gamma_min = -dri_dot_r
		else:
			gamma_min = l_i

		# Find delta_min
		if(dri_dot_r < 0):
			delta_min = 0
		elif(dri_dot_r >=0 and dri_dot_r < l_j):
			delta_min = dri_dot_r
		else:
			delta_min = l_j

		# Find gamma_max
		if(l_j - dri_dot_r < 0):
			gamma_max = 0.0
		elif(l_j - dri_dot_r >=0 and l_j - dri_dot_r < l_i):
			gamma_max = l_j - dri_dot_r
		else:
			gamma_max = l_i

		# Find delta_max:
		if(l_i + dri_dot_r < 0):
			delta_max = 0.0
		elif(l_i + dri_dot_r >= 0 and l_i + dri_dot_r < l_j):
			delta_max = l_i + dri_dot_r 
		else:
			delta_max  = l_j

		# Find gamma_final and delta_final by choosing the mid-point of the overlap if there is overlap
		if (gamma_min!=gamma_max and delta_min!= delta_max):
			# In case the two lines have an overlap region
			gamma_final = (gamma_min + gamma_max)/2.0
			delta_final = (delta_min + delta_max)/2.0
		else:
			# In the case where there is no overlap.
			# In this case the line of closest approach passes through one of the end-points in each line.
			gamma_final = gamma_max
			delta_final = delta_max

		return gamma_final, delta_final
		
		

		

