cimport cython
import numpy as np
from libc.math cimport sqrt
from cython.parallel import prange
cdef double PI = 3.14159265359

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)

cdef class filament:
    def __init__(self, a, Np, eta): 
        self.a = a 
        self.Np = Np
        self.eta = eta

    	# calculate the pair-wise separation vector
	def get_separation_vector(self):
		# length Np-1
		self.dx = self.r[1:self.Np] - self.r[0:self.Np-1]
		self.dy = self.r[self.Np+1:2*self.Np] - self.r[self.Np:2*self.Np-1]
		self.dz = self.r[2*self.Np+1:3*self.Np] - self.r[2*self.Np:3*self.Np-1]
		
		
		# length Np-1
		# Lengths of the separation vectors
		self.dr = (self.dx**2 + self.dy**2 + self.dz**2)**(1/2)
		
		# length Np-1
		self.dx_hat = self.dx/self.dr
		self.dy_hat = self.dy/self.dr
		self.dz_hat = self.dz/self.dr
		
		# rows: dimensions, columns : particles
		# Shape : dim x Np-1
		# Unit separation vectors 
		self.dr_hat = np.vstack((self.dx_hat, self.dy_hat, self.dz_hat))


    # Calculate bond-angle vector for the filament
	cpdef get_bound_angles(self, double[:] dr_hat, double [:] cosAngle):
		
		cdef int ii, jj, Np = self.Np, dim = self.dim

		unit_vector_x = np.array([1,0,0], dtype = 'int')

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

		
		for ii in range(self.Np-1):
			
			# For the boundary-points, store the angle wrt to the x-axis of the global cartesian coordinate system
			if(ii==0 or ii == self.Np-1):
				
				self.cosAngle[ii] = np.dot(self.dr_hat[:,ii], [1, 0 , 0])
				
			else:
				self.cosAngle[ii] = np.dot(self.dr_hat[:,ii-1], self.dr_hat[:,ii] )
				
		
#        print(self.cosAngle)
		
	# Find the local tangent vector of the filament at the position of each particle
	def getTangentVectors(self):
		
		# Unit tangent vector at the particle locations
		self.t_hat = np.zeros((self.dim,self.Np))
		
		for ii in range(self.Np):
			
			if ii==0:
				self.t_hat[:,ii] = self.dr_hat[:,ii]
			elif ii==self.Np-1:
				self.t_hat[:,-1] = self.dr_hat[:,-1]
			else:
				# For particles that have a neighbhor on each side, the tangent vector is the average of the two bonds. 
				vector = self.dr_hat[:,ii-1] + self.dr_hat[:,ii]
				self.t_hat[:,ii] = vector/(np.dot(vector, vector)**(1/2))
				
		
		self.t_hat_array = self.reshapeToArray(self.t_hat)
		
		# Initialize the particle orientations to be along the local tangent vector
		self.p = self.t_hat_array
				
	


	def BendingForces(self):
		# See notes for derivation of these expressions for bending potentials.

		self.getBondAngles()
		
		self.F_bending = np.zeros((self.dim,self.Np))
		
		for ii in range(self.Np):
			
			if ii==0:
				# Torque-free ends
				self.F_bending[:,ii] = self.kappa_array[ii+1]*(1/self.dr[ii])*(self.dr_hat[:, ii]*self.cosAngle[ii+1] - self.dr_hat[:, ii+1])
				
			elif ii == self.Np-1:
				# Torque-free ends
				self.F_bending[:,ii] = self.kappa_array[ii-1]*(1/self.dr[ii-1])*(self.dr_hat[:, ii-2] - self.cosAngle[ii - 1]*self.dr_hat[:, ii-1])
				
			else:
				
				if(ii!=1):
					term_n_minus = self.kappa_array[ii-1]*(self.dr_hat[:, ii-2] - self.dr_hat[:, ii-1]*self.cosAngle[ii-1])*(1/self.dr[ii-1])
				else:
					term_n_minus = 0
					
				term_n1 = (1/self.dr[ii-1] + self.cosAngle[ii]/self.dr[ii])*self.dr_hat[:, ii]
				term_n2 = -(1/self.dr[ii] + self.cosAngle[ii]/self.dr[ii-1])*self.dr_hat[:, ii-1]
				
				term_n = self.kappa_array[ii]*(term_n1 + term_n2)
				
				if(ii!=self.Np-2):
					term_n_plus = self.kappa_array[ii+1]*(-self.dr_hat[:, ii+1] + self.dr_hat[:, ii]*self.cosAngle[ii + 1])*(1/self.dr[ii])
				else:
					term_n_plus = 0
					
				self.F_bending[:,ii] =  term_n_minus + term_n + term_n_plus
				
			
		# Now reshape the forces array
		self.F_bending_array = self.reshapeToArray(self.F_bending)  
 
	
	def ConnectionForces(self):
	
#        def int Np = self.Np, i, j, xx = 2*Np
#        def double dx, dy, dz, dr2, dr, idr, fx, fy, fz, fac
		xx = 2*self.Np
		self.F_conn = np.zeros(self.dim*self.Np)
		
		for i in range(self.Np):
			fx = 0.0; fy = 0.0; fz = 0.0;
			for j in range(i,self.Np):
				
				if((i-j)==1 or (i-j)==-1):
					
					dx = self.r[i   ] - self.r[j   ]
					dy = self.r[i+self.Np] - self.r[j+self.Np]
					dz = self.r[i+xx] - self.r[j+xx] 
					dr2 = dx*dx + dy*dy + dz*dz
					dr = dr2**(1/2)
					
	#                    dr_hat = np.array([dx, dy, dz], dtype = 'float')*(1/dr)
					
					fac = -self.k*(dr - self.b0)
				
					fx = fac*dx/dr
					fy = fac*dy/dr
					fz = fac*dz/dr
					
					
					# Force on particle "i"
					self.F_conn[i]    += fx 
					self.F_conn[i+self.Np] += fy 
					self.F_conn[i+xx] += fz 
					
					# Force on particle "j"
					self.F_conn[j]    -= fx 
					self.F_conn[j+self.Np] -= fy 
					self.F_conn[j+xx] -= fz 
