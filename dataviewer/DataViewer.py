# Script to playback and interact with PyFilaments Data
import numpy as np
# from pyqtgraph.Qt import QtWidgets,QtCore, QtGui
import pyqtgraph as pg
pg.setConfigOptions(antialias=True)
import time as time
import os
import sys
import cmocean
import context
from pyfilaments.activeFilaments import activeFilament


os.environ["QT_API"] = "pyqt5"
import qtpy

# qt libraries
from qtpy.QtCore import *
from qtpy.QtWidgets import *
from qtpy.QtGui import *


class AnimatePlotWidget(pg.GraphicsLayoutWidget):

	def __init__(self, filament = None, parent=None):
		
		super().__init__(parent)

		self.plot = self.addPlot()

		self.plot.setAspectLocked(True)


		self.plot.setRange(xRange=(0, 150), yRange=(-150,150),disableAutoRange=False)

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
		
		print(self.filament.Np)

		for ii in range(self.filament.Np):
			
			if(self.filament.S_mag[ii]!=0 or self.filament.D_mag[ii]!=0):
				self.particle_colors.append(pg.mkBrush(255, 0, 0, 200))
			else:
				self.particle_colors.append(pg.mkBrush(0, 0, 255, 200))


class VideoPlayer(QWidget):
	'''
	A general class for displaying sequential data such as plots, images etc.

	'''

	def __init__(self, filament = None, plotFilamentWidget = None, parent=None):
		
		super().__init__(parent)

		# Data used for displaying the sequence of plots. 
		self.filament = filament

		self.plotFilamentWidget = plotFilamentWidget

		# Playback indices
		self.timer = QTimer()
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

		self.add_components()


		

	def add_components(self):

		# Play Button
		self.playButton = QPushButton()
		self.playButton.setEnabled(False)
		self.playButton.setCheckable(True)
		self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

		self.positionSlider = QSlider(Qt.Horizontal)
		self.positionSlider.setRange(0, 0)
		self.positionSlider.setEnabled(False)
		self.positionSlider_prevValue=0
		
		self.positionSpinBox = QDoubleSpinBox()
		self.positionSpinBox.setRange(0, 0)
		self.positionSpinBox.setSingleStep(0.01)
		self.positionSpinBox.setEnabled(False)
		self.positionSpinBox_prevValue=0
		
		controlLayout = QHBoxLayout()
		controlLayout.setContentsMargins(0, 0, 0, 0)
		
		controlLayout.addWidget(self.playButton)
		controlLayout.addWidget(self.positionSlider)
		controlLayout.addWidget(self.positionSpinBox)

		self.setLayout(controlLayout)
		

		# Make connections
		self.playButton.clicked.connect(self.play)
		self.positionSlider.valueChanged.connect(self.positionSpinBox_setValue)
		self.positionSpinBox.valueChanged.connect(self.positionSlider_setValue)


	def positionSpinBox_setValue(self,value):
	
		newvalue = self.filament.Time[value]
		self.positionSpinBox.setValue(newvalue)
		self.positionSlider_prevValue = value
		
	def positionSlider_setValue(self,value):
		
		newvalue, hasToChange = self.find_slider_index(value)
	   
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

			# print(self.plotFilamentWidget.current_index)
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









class simParamsDisplayWidget(QWidget):

	update_main_window = Signal()

	def __init__(self, filament = None, parent=None):
		
		super().__init__(parent)

		self.filament = filament

		self.displayed_parameters = ['N particles', 'radius', 'bond length', 'spring constant', 'kappa_hat', 'simulation type', 'force strength', 'stresslet strength',
									'potDipole strength']

		self.variable_mapping = {'N particles':self.filament.Np, 'radius':self.filament.radius, 'bond length':self.filament.b0, 'spring constant':self.filament.k, 
								'kappa_hat':self.filament.kappa_hat, 'simulation type': self.filament.sim_type, 'force strength':self.filament.F0, 
								'stresslet strength':self.filament.S0, 'potDipole strength':self.filament.D0}

		assert(list(self.variable_mapping.keys()) == self.displayed_parameters)

		self.add_components()
		

	def add_components(self):

		self.labels = {key:[] for key in self.displayed_parameters}
		self.value_labels = {key:[] for key in self.displayed_parameters}
		# Widget displaying simulation parameters
		for parameter in self.displayed_parameters:

			self.labels[parameter] = QLabel(parameter + ' : ')

			self.value_labels[parameter] = QLabel('')

			if(self.variable_mapping[parameter] is not None):
				self.value_labels[parameter].setText(str(self.variable_mapping[parameter]))
			else:
				self.value_labels[parameter].setText('')




		sim_params_display = QGridLayout()

		first_row = self.displayed_parameters[:6]
		second_row = self.displayed_parameters[7:]

		col_counter=0
		for col_no, parameter in enumerate(first_row):

			sim_params_display.addWidget(self.labels[parameter], 0, col_counter)
			col_counter+=1
			sim_params_display.addWidget(self.value_labels[parameter], 0, col_counter)
			col_counter+=1

		col_counter=0
		for col_no, parameter in enumerate(second_row):

			sim_params_display.addWidget(self.labels[parameter], 1, col_counter)
			col_counter+=1
			sim_params_display.addWidget(self.value_labels[parameter], 1, col_counter)
			col_counter+=1

		self.setLayout(sim_params_display)

		
 
	def update_param_values(self):

		print('Updating parameter values...')

		self.variable_mapping = {'N particles':self.filament.Np, 'radius':self.filament.radius, 'bond length':self.filament.b0, 'spring constant':self.filament.k, 
								'kappa_hat':self.filament.kappa_hat, 'simulation type': self.filament.sim_type, 'force strength':self.filament.F0, 
								'stresslet strength':self.filament.S0, 'potDipole strength':self.filament.D0}

		for parameter in self.displayed_parameters:


			if(self.variable_mapping[parameter] is not None):
				
				print(str(self.variable_mapping[parameter]))

				self.value_labels[parameter].setText(str(self.variable_mapping[parameter]))
			else:

				self.value_labels[parameter].setText('')


		QApplication.processEvents()


'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            Central Widget
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
class CentralWidget(QWidget):
	
	update_plot = Signal(float)

	index = Signal(int)
   
	def __init__(self):
		super().__init__()

		
		self.filament = activeFilament()

		self.newData = False

		# Widget for displaying the filament as a scatter plot
		self.plotFilamentWidget = AnimatePlotWidget(filament = self.filament)

		# Widget for displaying the simulation parameters
		self.sim_parameters_display = simParamsDisplayWidget(filament = self.filament)

		# Widget for sequentially displaying data
		self.video_player = VideoPlayer(filament = self.filament, plotFilamentWidget = self.plotFilamentWidget)

	
		# Add widgets to the central widget

		window_layout = QVBoxLayout()

		window_layout.addWidget(self.plotFilamentWidget)
		window_layout.addWidget(self.video_player)
		window_layout.addWidget(self.sim_parameters_display)

		self.setLayout(window_layout)

		# if(self.newData is True):

		# 	self.video_player.initialize_data(self.filament.Time)

		# 	self.video_player.initialize_parameters()

		# 	self.sim_parameters_display.update_param_values()

			

	def open_dataset(self, fileName):

		print(self.filament.Np)

		self.fileName = fileName
		
		self.filament.load_data(self.fileName)

		print('Loaded data successfully!')


		if(self.filament.R is not None):
			self.newData = True

			self.plotFilamentWidget.set_colors()
			self.video_player.initialize_data(self.filament.Time)
			self.video_player.initialize_parameters()
			self.sim_parameters_display.update_param_values()

		
		else:
			self.newData = False



	
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            Main Window
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
		
class MainWindow(QMainWindow):
	
   
	def __init__(self):
		super().__init__()
		
		self.setWindowTitle('Active filaments data analyzer')
		# self.setWindowIcon(QtGui.QIcon('icon/icon.png'))
		self.statusBar().showMessage('Ready')
		

		
		
		#WIDGETS
		self.central_widget = CentralWidget()  
		self.setCentralWidget(self.central_widget)


		# File and Folder
		self.dataFile = None
		# self.dataFile = '/Users/deepak/Dropbox/LacryModeling/ModellingResults/SimResults_Np_32_Shape_sinusoid_k_1_b0_4_S_0_D_-1/SimResults_Tmax_5000_Np_32_Shape_sinusoid_S_0_D_-1.pkl'
		self.directory = '/Users/deepak/Dropbox/LacryModeling/ModellingResults'

		# Create menu bar and add action
		menuBar = self.menuBar()
		fileMenu = menuBar.addMenu('&File')
	 
		
		# Create new action
		openAction = QAction(QIcon('open.png'), '&Open', self)        
		openAction.setShortcut('Ctrl+O')
		openAction.setStatusTip('Open File')
		openAction.triggered.connect(self.openFile)

		fileMenu.addAction(openAction)


	def openFile(self):
		print('Opening dataset ...')

		# if(self.dataFile is None):
			# self.directory = QtGui.QFileDialog.getExistingDirectory(self)

		# file_type = "pkl files (*.pkl)"
		

		self.dataFile, *rest = QFileDialog.getOpenFileName(self, 'Open file',self.directory,"data files (*.pkl *.hdf5)")

		self.central_widget.open_dataset(self.dataFile)



'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
							 Main Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

if __name__ == '__main__':

	#To prevent the error "Kernel died"
	
	app = QApplication.instance()
	if app is None:
		app = QApplication(sys.argv)
	
	win= MainWindow()
	# qss = QSSHelper.open_qss(os.path.join('aqua', 'aqua.qss'))
	# win.setStyleSheet(qss)


		
	win.show()

	app.exec_() #

	
	# if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
	# 	QApplication.instance().exec_()