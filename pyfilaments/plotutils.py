# Display functions for active filaments
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt 
import pyfilaments.activeFilaments


def plotFilament(r = None):
		
	
		ax1 = plt.gca()
		
#        1ax = fig.add_subplot(1,1,1)
		

		ax1.scatter(r[:self.Np], r[self.Np:2*self.Np], 20, c = self.particle_colors, alpha = 0.75, zorder = 20, cmap = cmocean.cm.curl)
		ax1.plot(r[:self.Np], r[self.Np:2*self.Np], color = 'k', alpha = 0.5, zorder = 10)

#        ax.set_xlim([-0.1, self.Np*self.b0])
#        ax.set_ylim([-self.Np*self.b0/2, self.Np*self.b0/2])
		
#        fig.canvas.draw()
#    
	def plotSimResult(save=False):
		
		if(save):
			self.FilImages = os.path.join(self.saveFolder, 'FilamentImages_Np_{}_Shape_{}_S_{}_D_{}'.format(self.Np, self.shape, self.S0, self.D0))      
#        
			if(not os.path.exists(self.FilImages)):
	#            
				os.makedirs(self.FilImages)
			
			
		fig = plt.figure()
		
		ax = fig.add_subplot(1,1,1)
#        ax1 = fig.add_axes([0.85,0.05,0.02,0.85])
#        cax1 = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)

		
		if(self.R is not None):
			
			TimePts, rest  = np.shape(self.R)
			
			COM_array = np.zeros((self.dim, TimePts))
			COM_Speed = np.zeros(TimePts)
			
			self.x_min = np.min(self.R[:,:self.Np])
			self.x_max = np.max(self.R[:,:self.Np])
			
			self.y_min = np.min(self.R[:,self.Np:2*self.Np])
			self.y_max = np.max(self.R[:,self.Np:2*self.Np])
			
			
			for ii in range(TimePts):
				
				R = self.R[ii,:]
				t = self.Time[ii]
				
				COM_array[0,ii] = np.nanmean(R[:self.Np])
				COM_array[1,ii] = np.nanmean(R[self.Np:2*self.Np])
				
				if(ii>0):
					COM_Speed[ii] = ((COM_array[0,ii] - COM_array[0,ii-1])**2 + (COM_array[1,ii] - COM_array[1,ii-1])**2 + (COM_array[2,ii] - COM_array[2,ii-1])**2)**(1/2)
				
				ax.clear()
								   
								
				im = ax.scatter(COM_array[0,:ii], COM_array[1,:ii], 100, c = COM_Speed[:ii], alpha = 1.0, zorder = 2, cmap = 'viridis')
				
				self.plotFilament(self.r)                
#                cbar = fig.colorbar(im, cax = cax1, orientation = 'vertical')
				ax.set_title('Time: {:.2f}'.format(t))
				
				ax.set_xlabel('X')
				ax.set_ylabel('Y')
#                cbar.set_label('COM speed')
				ax.axis('equal')
#                ax.axis([self.x_min - 2*self.radius, self.x_max + 2*self.radius , self.y_min-2*self.radius, self.y_max +2*(self.radius)])
#                ax.set_xlim([x_min - 2*self.radius, x_max + 2*self.radius])
#                ax.set_ylim([y_min-2*self.radius, y_max +2*(self.radius)])
				
				if(save):
					plt.savefig(os.path.join(self.FilImages, 'Res_T_{:.2f}_'.format(t)+'.tif'), dpi = 150)
#                cbar = plt.colorbar(ax)
				plt.pause(0.000001)
				fig.canvas.draw()

	def plotFlowFields(self, save = False):
		# Plots the fluid-flow field around the filament at different times of the simulation
		
		Ng = 32
		Nt = Ng*Ng
		
		rt = np.zeros(dim*Nt)                   # Memory Allocation for field points
		vv = np.zeros(dim*Nt)                   # Memory Allocation for field Velocities
		
		if(save):
			self.flowFolder = os.path.join(self.saveFolder, 'FlowFields_Np_{}_Shape_{}_S_{}_D_{}'.format(self.Np, self.shape, self.S0, self.D0))      
			if(not os.path.exists(self.flowFolder)):      
				os.makedirs(self.flowFolder)
		
		
		if(self.R is not None):
			TimePts, rest  = np.shape(self.R)
			
			COM_array = np.zeros((self.dim, TimePts))
			COM_Speed = np.zeros(TimePts)
			
#            self.x_min = np.min(self.R[:,:self.Np]) - 5*self.radius
#            self.x_max = np.max(self.R[:,:self.Np]) + 5*self.radius
#            
#            self.y_min = np.min(self.R[:,self.Np:2*self.Np]) - 5*self.radius
#            self.y_max = np.max(self.R[:,self.Np:2*self.Np]) + 5*self.radius
			
			
			ii = int(TimePts/2)
				
			R = self.R[ii,:]
			t = self.Time[ii]
			
			self.r = R
			
			COM_array[0,ii] = np.nanmean(R[:self.Np])
			COM_array[1,ii] = np.nanmean(R[self.Np:2*self.Np])
			
			self.x_min = COM_array[0,ii] - self.L/2 - 20*self.radius
			self.x_max = COM_array[0,ii] + self.L/2 + 20*self.radius
			self.y_min = COM_array[1,ii] - self.L/2 - 20*self.radius
			self.y_max = COM_array[1,ii] + self.L/2 + 20*self.radius
			
			
			
			
			# Define a grid for calculating the velocity vectors based on the above limits
			# creating a meshgrid
			xx = np.linspace(self.x_min, self.x_max, Ng)
			yy = np.linspace(self.y_min, self.y_max, Ng)
			X, Y = np.meshgrid(xx, yy)
			rt[0:2*Nt] = np.concatenate((X.reshape(Ng*Ng), Y.reshape(Ng*Ng)))
			
			####Instantiate the Flow class
			uFlow = pystokes.unbounded.Flow(self.radius, self.Np, self.eta, Nt)
			
			plt.figure(1)
			
			for ii in range(TimePts):
			
#            ii = int(TimePts/2)
				
				R = self.R[ii,:]
				t = self.Time[ii]
				
				self.r = R
				
			
				COM_array[0,ii] = np.nanmean(R[:self.Np])
				COM_array[1,ii] = np.nanmean(R[self.Np:2*self.Np])
				
				# Uncomment below to plot the flow-fields at a fixed point
				self.x_min = COM_array[0,ii] - self.L - 50*self.radius
				self.x_max = COM_array[0,ii] + self.L + 50*self.radius
				self.y_min = COM_array[1,ii] - self.L - 50*self.radius
				self.y_max = COM_array[1,ii] + self.L + 50*self.radius
				
				# Define a grid for calculating the velocity vectors based on the above limits
				# creating a meshgrid
				xx = np.linspace(self.x_min, self.x_max, Ng)
				yy = np.linspace(self.y_min, self.y_max, Ng)
				X, Y = np.meshgrid(xx, yy)
				rt[0:2*Nt] = np.concatenate((X.reshape(Ng*Ng), Y.reshape(Ng*Ng)))
				
				
				# Calculate the forces on the particles due to intrinsic forces
				self.getSeparationVector()
				self.getBondAngles()
				self.getTangentVectors()
			
				self.setStresslet()
			
				self.calcForces()
				
				
				
				uFlow.stokesletV(vv, rt, R, self.F) # computes flow (vv) at the location of rt in vector vv given r and F
				
				uFlow.stressletV(vv, rt, R, self.S) # computes flow (vv) at the location of rt in vector vv given r and S

				
				vx, vy = vv[0:Nt].reshape(Ng, Ng), vv[Nt:2*Nt].reshape(Ng, Ng)
	
				U = (vx**2 + vy**2)**(1/2)
				U = np.log(U/np.nanmax(U))
				
				
			
				# Plot
			   
				
				plt.clf()
				
				self.plotFilament(self.r)
				
				ax2 = plt.contourf(X, Y, U, 20, cmap = cmocean.cm.amp, alpha=1.0,linewidth=0,linestyle=None, zorder = 0)
				
	
	
				plt.streamplot(X, Y, vx, vy, color="black", density=1, linewidth =1, arrowsize = 1, zorder = 1)
#                plt.quiver(X, Y, vx, vy, color="black", linewidth =1)
				im = plt.scatter(COM_array[0,:ii], COM_array[1,:ii], 25, c = COM_Speed[:ii], alpha = 0.75, zorder = 2, cmap = 'viridis')

	
				cbar = plt.colorbar(ax2)
	#
				cbar.set_label(r'$log(U/U_{max})$', fontsize =16)
				
				plt.axis('equal')
				plt.xlim([self.x_min, self.x_max])
				plt.ylim([self.y_min, self.y_max])
				plt.xlabel(r'$x/a$', fontsize=16)
				plt.ylabel(r'$y/a$', fontsize=16)
				plt.title('Time = {:.2f}'.format(t))

	#            plt.axes(aspect = 'equal')
				
	
				if(save):
					plt.savefig(os.path.join(self.flowFolder, 'FlowField_T_{:.2f}_'.format(t)+'.png'), dpi = 150)

				plt.pause(0.00001)
				plt.show(block = False)

	def plotFilamentStrain(self):

		if(self.R is not None):

			TimePts, rest  = np.shape(self.R)

			strain_vector = np.zeros((TimePts, self.Np - 1))

			pos_vector = np.zeros((self.dim, self.Np))
			
			for ii in range(TimePts):

				pos_vector[0,:] = self.R[ii, :self.Np]
				pos_vector[1,:] = self.R[ii, self.Np:2*self.Np]
				pos_vector[2,:] = self.R[ii, 2*self.Np:3*self.Np]

				diff = np.diff(pos_vector, axis = 1)

				link_distance = (diff[0,:]**2 + diff[1,:]**2 + diff[2,:]**2)**(1/2)

				strain_vector[ii,:] = link_distance/self.b0




			# Plot the initial and final strain vectors

			plt.figure()
			plt.plot(range(self.Np-1), strain_vector[0,:], 'ro', label = 'Initial')
			plt.plot(range(self.Np-1), strain_vector[-1,:], 'ro', label = 'Final')
			plt.xlabel('Link number')
			plt.ylabel('Strain')
			plt.show()

