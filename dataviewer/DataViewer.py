# Script to playback and interact with PyFilaments Data
import numpy as np
from pyqtgraph.Qt import QtWidgets,QtCore, QtGui
import pyqtgraph as pg
pg.setConfigOptions(antialias=True)
import time as time
import os
import sys
import cmocean
import context
from pyfilaments.activeFilaments import activeFilament


# fileName = ''

# filament = activeFilament()

# filament.loadData(fileName)

class AnimatePlotWidget(pg.GraphicsLayoutWidget):

	def __init__(self, filament = None, parent=None):
		
		super().__init__(parent)

		self.plot = self.addPlot()

		self.plot.setAspectLocked(True)


		self.plot.setRange(xRange=(0, 150), yRange=(-150,150),disableAutoRange=True)

		self.filament = filament

		

		self.current_index = 0


		if(self.filament.R is None):
			n = 1

			self.s1 = pg.ScatterPlotItem(size=10, pen = pg.mkPen(None), brush = pg.mkBrush(255, 255, 255, 120))

			self.s2 = pg.ScatterPlotItem(size=10, pen = pg.mkPen(None), brush = pg.mkBrush(255, 255, 255, 120))


			pos = np.random.normal(size = (2,n))

			# spots = [{'pos': pos[:,i], 'data': 1} for i in range(n)] + [{'pos': [0,0], 'data': 1}]
			
			# s1.addPoints(spots)

			self.s1.setData(x = pos[0,:], y = pos[1,:], brush = pg.mkBrush(255, 0, 255, 200))

		else:
			self.s1 = pg.ScatterPlotItem(size=10, pen = pg.mkPen(None), brush = pg.mkBrush(255, 0, 255, 200))

			self.s2 = pg.ScatterPlotItem(size=10, pen = pg.mkPen(None), brush = pg.mkBrush(255, 0, 255, 200))


			x_pos = self.filament.R[self.current_index,:self.filament.Np]
			y_pos = self.filament.R[self.current_index,self.filament.Np:2*self.filament.Np]

			# spots = [{'pos': pos[:,i], 'data': 1} for i in range(n)] + [{'pos': [0,0], 'data': 1}]
			
			# s1.addPoints(spots)

			self.s1.setData(x = x_pos, y = y_pos)

			self.s1.setBrush(self.particle_colors)

		
		self.plot.addItem(self.s1)
		self.plot.addItem(self.s2)





	def update_plot(self):

		x_pos = self.filament.R[self.current_index,:self.filament.Np]
		y_pos = self.filament.R[self.current_index,self.filament.Np:2*self.filament.Np]
		z_pos = self.filament.R[self.current_index, 2*self.filament.Np : 3*self.filament.Np]

		print(z_pos)

		self.s1.setData(x = x_pos, y = y_pos)
		self.s1.setBrush(self.particle_colors)

		# Display head position
		# x_pos_head = self.filament.R[:self.current_index,self.filament.Np-1]
		# y_pos_head = self.filament.R[:self.current_index,2*self.filament.Np-1]

		# self.s2.setData(x = x_pos_head, y = y_pos_head, brush = pg.mkBrush(255, 0, 0, 120))





	def set_colors(self):
		self.particle_colors = []
		self.passive_color = np.reshape(np.array(cmocean.cm.curl(0)),(4,1))
		self.active_color =np.reshape(np.array(cmocean.cm.curl(255)), (4,1))
		
		for ii in range(self.filament.Np):
			
			if(self.filament.S_mag[ii]!=0 or self.filament.D_mag[ii]!=0):
				self.particle_colors.append(pg.mkBrush(255, 0, 0, 200))
			else:
				self.particle_colors.append(pg.mkBrush(0, 0, 255, 200))






		





'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            Central Widget
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
class CentralWidget(QtWidgets.QWidget):
	update_plot=QtCore.pyqtSignal(float)

	index = QtCore.pyqtSignal(int)
   
	def __init__(self):
		super().__init__()

		
		self.filament = activeFilament()

		self.newData = False

		# Widget for displaying the filament as a scatter plot
		self.plotFilamentWidget = AnimatePlotWidget(filament = self.filament)


		# Playback indices
		self.timer = QtCore.QTimer()
		self.timer.setInterval(0) #in ms
		self.timer.timeout.connect(self.play_refresh)
		self.current_track_time=0
		self.current_computer_time=0
		
		self.current_track_index = 0        # Add image index
		self.prev_track_index = 0  

		# If true then plays the data in real-time
		self.real_time = False
		# No:of frames to advance for recording purposes
		self.frames = 1
		# This gives playback_speed x normal speed
		self.playback_speed = 200
		#Gui Component


		# Play Button
		self.playButton = QtGui.QPushButton()
		self.playButton.setEnabled(False)
		self.playButton.setCheckable(True)
		self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
		self.playButton.clicked.connect(self.play)

		self.positionSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
		self.positionSlider.setRange(0, 0)
		self.positionSlider.setEnabled(False)
		self.positionSlider_prevValue=0
		
		self.positionSpinBox=QtGui.QDoubleSpinBox()
		self.positionSpinBox.setRange(0, 0)
		self.positionSpinBox.setSingleStep(0.01)
		self.positionSpinBox.setEnabled(False)
		self.positionSpinBox_prevValue=0
		
		self.positionSlider.valueChanged.connect(self.positionSpinBox_setValue)
		self.positionSpinBox.valueChanged.connect(self.positionSlider_setValue)

		# Create layouts to place inside widget
		playBack_Controls = QtWidgets.QWidget()
		controlLayout = QtGui.QHBoxLayout()
		controlLayout.setContentsMargins(0, 0, 0, 0)
		
		controlLayout.addWidget(self.playButton)
		controlLayout.addWidget(self.positionSlider)
		controlLayout.addWidget(self.positionSpinBox)

		playBack_Controls.setLayout(controlLayout)

		window_layout = QtGui.QVBoxLayout()

		window_layout.addWidget(self.plotFilamentWidget)
		window_layout.addWidget(playBack_Controls)

		self.setLayout(window_layout)

		if(self.newData is True):

			self.initialize_data(self.filament.Time)

			self.initialize_parameters()

			

	def openDataset(self, fileName):

		

		self.fileName = fileName
		
		self.filament.loadData(self.fileName)



		if(self.filament.R is not None):
			self.newData = True

			self.initialize_data(self.filament.Time)

			self.initialize_parameters()

			self.plotFilamentWidget.set_colors()

		


		else:
			self.newData = False



	def positionSpinBox_setValue(self,value):
	
		newvalue=self.filament.Time[value]
		self.positionSpinBox.setValue(newvalue)
		self.positionSlider_prevValue=value
		
	def positionSlider_setValue(self,value):
		
		newvalue, hasToChange=self.find_slider_index(value)
	   
		self.current_track_index = newvalue
		if hasToChange:
			self.positionSlider.setValue(newvalue)
			self.positionSpinBox.setValue(self.Time[newvalue])
			self.positionSpinBox_prevValue=self.Time[newvalue]
			
#            self.prev_track_index = newvalue
#            self.prev_track_index = self.Image_Index[newvalue]
			
			self.positionSlider_prevValue=newvalue
			# self.refreshImage(self.Image_Names[newvalue])
			self.plotFilamentWidget.current_index = newvalue

			print(self.plotFilamentWidget.current_index)
			self.plotFilamentWidget.update_plot()
			# self.update_3Dplot.emit(self.Image_Time[newvalue])

	def find_slider_index(self, value):
		#
		index=0
		found=False
		hasToChange=True
		
		if self.positionSpinBox_prevValue<value: 
			for i in range(0,len(self.Time)):
				if self.filament.Time[i]-value>=0 and not(found):
					index=i
					found=True
			if not(found):
				index=len(self.Time)-1
				
		elif self.positionSpinBox_prevValue>value:
			for i in range(len(self.Time)-1,-1,-1):
				if self.Time[i]-value<=0 and not(found):
					index=i
					found=True
			if not(found):
				index=0
		else:
			hasToChange=False
			
		return index,hasToChange
	
	def initialize_parameters(self):
		self.playButton.setEnabled(True) 

		if self.playButton.isChecked():
			self.playButton.setChecked(False)
			self.timer.stop()
			self.positionSlider.setEnabled(True)
			self.positionSpinBox.setEnabled(True)
			
		self.current_track_index = 0
		self.current_track_time=0
		self.current_computer_time=0
		self.positionSlider.setValue(0)
		self.positionSpinBox_prevValue=0
		self.positionSlider_prevValue=0

	def initialize_data(self,time):
		self.Time = time
		self.positionSlider.setRange(0,len(self.Time)-1)
		self.positionSlider.setEnabled(True)
		self.positionSpinBox.setRange(self.Time[0],self.Time[-1])
		self.positionSpinBox.setEnabled(True)

	# Function that updates the plots
	def play(self):
		if self.playButton.isChecked():
			self.current_computer_time = time.time()
			self.current_track_time = self.positionSpinBox_prevValue
			# Set the prev index to the current index
			self.prev_track_index = self.current_track_index
			self.timer.start(0)
			self.positionSlider.setEnabled(False)
			self.positionSpinBox.setEnabled(False)
		else:
			self.timer.stop()
			self.positionSlider.setEnabled(True)
			self.positionSpinBox.setEnabled(True)

	def play_refresh(self):
		
		if self.real_time:
			timediff = time.time()-self.current_computer_time


			timediff_scaled = self.playback_speed*timediff
			
			index = np.argmin(abs(self.Time-(timediff_scaled+self.current_track_time)))

			
			if index>self.positionSlider_prevValue:
				self.current_computer_time += timediff
				self.current_track_time += timediff_scaled
				self.positionSlider.setValue(index)
				
		else:
			
			self.current_track_index = self.prev_track_index + self.frames
		
			self.positionSlider.setValue(self.current_track_index)
			
			self.prev_track_index = self.current_track_index

			# Loop the timer if the end is reached.
			if(self.current_track_index == len(self.Time)):
				self.current_track_index = 0
				self.prev_track_index = 0
	
	def connect_all(self):
		pass
	
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            Main Window
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
		
class MainWindow(QtWidgets.QMainWindow):
	
   
	def __init__(self):
		super().__init__()
		
		self.setWindowTitle('Active filaments data analyzer')
		# self.setWindowIcon(QtGui.QIcon('icon/icon.png'))
		self.statusBar().showMessage('Ready')
		

		
		
		#WIDGETS
		self.central_widget=CentralWidget()  
		self.setCentralWidget(self.central_widget)


		# File and Folder
		self.dataFile = None
		# self.dataFile = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_32_Shape_sinusoid_k_1_b0_4_S_0_D_-1/SimResults_Tmax_5000_Np_32_Shape_sinusoid_S_0_D_-1.pkl'
		self.directory = '/Users/deepak/Dropbox/LacryModeling/ModellingResults'

		# Create menu bar and add action
		menuBar = self.menuBar()
		fileMenu = menuBar.addMenu('&File')
	 
		
		# Create new action
		openAction = QtGui.QAction(QtGui.QIcon('open.png'), '&Open', self)        
		openAction.setShortcut('Ctrl+O')
		openAction.setStatusTip('Open File')
		openAction.triggered.connect(self.openFile)

		fileMenu.addAction(openAction)


	def openFile(self):
		print('Opening dataset ...')

		# if(self.dataFile is None):
			# self.directory = QtGui.QFileDialog.getExistingDirectory(self)

		self.dataFile, *rest = QtGui.QFileDialog.getOpenFileName(self, 'Open file',self.directory,"pkl files (*.pkl)")

		self.central_widget.openDataset(self.dataFile)


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
							 Main Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

if __name__ == '__main__':

	#To prevent the error "Kernel died"
	
	app = QtGui.QApplication.instance()
	if app is None:
		app = QtGui.QApplication(sys.argv)
	
	win= MainWindow()
	# qss = QSSHelper.open_qss(os.path.join('aqua', 'aqua.qss'))
	# win.setStyleSheet(qss)
	win.central_widget.connect_all()

		
	win.show()

	

	
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_()