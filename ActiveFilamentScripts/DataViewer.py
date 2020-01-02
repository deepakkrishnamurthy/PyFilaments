# Script to playback and interact with PyFilaments Data
import numpy as np
from pyqtgraph.Qt import QtWidgets,QtCore, QtGui
import pyqtgraph as pg
import time as time
import os
import sys
from Filament import activeFilament

# fileName = ''

# filament = activeFilament()

# filament.loadData(fileName)

class plotFilamentWidget(pg.GraphicsLayoutWidget):

	def __init__(self, parent=None):
		
		super().__init__(parent)

		self.plot = self.addPlot()

		n = 300

		s1 = pg.ScatterPlotItem(size=10, pen = pg.mkPen(None), brush = pg.mkBrush(255, 255, 255, 120))

		pos = np.random.normal(size = (2,n))

		# spots = [{'pos': pos[:,i], 'data': 1} for i in range(n)] + [{'pos': [0,0], 'data': 1}]
		
		# s1.addPoints(spots)

		s1.setData(x = pos[0,:], y = pos[1,:], brush = pg.mkBrush(255, 0, 255, 200))
		
		self.plot.addItem(s1)

	def update_plot(self):
		pass




'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            Central Widget
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
class CentralWidget(QtWidgets.QWidget):
	
   
	def __init__(self):
		super().__init__()

		self.filament = activeFilament()
		# Widget for displaying the filament as a scatter plot
		self.scatter_plot = plotFilamentWidget()

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

		window_layout.addWidget(self.scatter_plot)
		window_layout.addWidget(playBack_Controls)

		self.setLayout(window_layout)

	def positionSpinBox_setValue(self,value):
		pass
		# newvalue=self.Image_Time[value]
		# self.positionSpinBox.setValue(newvalue)
		# self.positionSlider_prevValue=value
		
	def positionSlider_setValue(self,value):
		pass
#         newvalue, hasToChange=self.find_slider_index(value)
	   
#         self.current_track_index = newvalue
#         if hasToChange:
#             self.positionSlider.setValue(newvalue)
#             self.positionSpinBox.setValue(self.Image_Time[newvalue])
#             self.positionSpinBox_prevValue=self.Image_Time[newvalue]
			
# #            self.prev_track_index = newvalue
# #            self.prev_track_index = self.Image_Index[newvalue]
			
#             self.positionSlider_prevValue=newvalue
#             self.refreshImage(self.Image_Names[newvalue])
#             self.update_plot.emit(self.Image_Time[newvalue])
#             self.update_3Dplot.emit(self.Image_Time[newvalue])
	# Function that updates the plots
	def play(self):
		pass
	
	def connect_all(self):
		pass
	


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
	
	central_widget = CentralWidget()
	mw = QtGui.QMainWindow()
	mw.setCentralWidget(central_widget)
	mw.setWindowTitle('Active Filament Data Viewer')
	mw.show()

	
	
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_()