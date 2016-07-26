import numpy as np
#from matplotlib import pyplot
import xml.etree.ElementTree as ET
from PyQt5.QtCore import QObject, pyqtSlot, pyqtProperty, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QTransform, QColor, QPainter
from PyQt5.QtWidgets import QApplication
from PyQt5.QtQuick import QQuickImageProvider, QQuickPaintedItem
from PyQt5.QtQml import qmlRegisterType
from scipy import signal

def parse_config(filename, rotate=True):
	tree = ET.parse(filename)
	root = tree.getroot()

	class Object(): pass
	config = Object()
	
	width = int( root.find(".//*[Name='Image Width']/Val").text )
	height = int( root.find(".//*[Name='Image Height']/Val").text )
	if rotate:
		config.image_width = height
		config.image_height = width
	else:
		config.image_width = width
		config.image_height = height
	#nframes = int( root.find(".//*[Name='Frames to read']/Val").text )
	config.framerate = float( root.find(".//*[Name='Frames per second']/Val").text )
	config.analog_samplerate = float( root.find(".//*[Name='Sample Rate (S/s)']/Val").text )
	
	return config

def get_video_nframes(filename):
	infile = open(filename)
	header = np.fromfile(infile, dtype='>H', count=2)
	framesize = header[1]*8 + 4
	
	infile.seek(0, 2)
	end = infile.tell()
	infile.close()
	return (end+1) // framesize

def read_frame(filename, image_width, image_height, frameindex, rotate=True):
	infile = open(filename)
	infile.seek(frameindex*(image_width*image_height+4) + 4)
	
	int64_data = np.fromfile(infile, dtype='>i8', count=image_height*image_width//8)
	infile.close()
	
	# data byte order in file: 7 6 5 4 3 2 1 0 15 14 13 12 11 10 9 8 23 22 ...
	# shuffle bytes to the right order
	datastr = int64_data.astype('<i8').tobytes()
	if rotate:
		bytearr = np.fromstring(datastr, dtype='uint8').reshape([image_width, image_height])
		return np.fliplr(bytearr.transpose()).tobytes()
	else:
		return datastr
	

def read_section(filename, xstart, ystart, tstart, xwindow, ywindow, twindow, image_width, image_height, rotate=True):
	if rotate:
		xstart, ystart = (ystart, image_width - xstart - xwindow)
		xwindow, ywindow = ywindow, xwindow
		image_width, image_height = image_height, image_width
		
	xstart -= xstart%8
	xwindow -= xwindow%8
	
	infile = open(filename)
	# 4 header bytes in each frame
	header = np.fromfile(infile, dtype='>H', count=2)
	
	int64_data = np.zeros([twindow, ywindow, xwindow//8], dtype='>i8')
	
	for t in range(twindow):
		infile.seek(4 + (tstart+t) * (image_width*image_height + 4) + ystart*image_width)
		datablock = np.fromfile(infile, dtype='>i8', count=ywindow*image_width//8)
		datablock = datablock.reshape([ywindow, image_width//8])
		int64_data[t,:,:] = datablock[:, xstart//8:(xstart+xwindow)//8]
	
	infile.close()
	
	# data byte order in file: 7 6 5 4 3 2 1 0 15 14 13 12 11 10 9 8 23 22 ...
	# shuffle bytes to the right order
	bytesdata = int64_data.astype('<i8').tobytes()
	result = np.fromstring(bytesdata, dtype='uint8').reshape([twindow, ywindow, xwindow])
	
	if rotate:
		result = np.fliplr(result).transpose([0, 2, 1]) # why in this order :P ??
	
	return result
	

def spectrogram(section, nperseg, step, framerate=1):
	nframes, ny, nx = section.shape
	ntransforms = (nframes-nperseg)//step
	nperseg += nperseg%2
	window = signal.get_window(('tukey', 0.25), nperseg)
	
	result = np.zeros([ntransforms, nperseg//2 + 1, ny, nx], dtype="float32")
	
	# shape: (x, y, t)
	transposed = section.transpose()
	
	for i in range(ntransforms):
		segment = transposed[:,:, i*step:i*step+nperseg]
		segment = signal.detrend(segment) * window
		temp = np.fft.rfft(segment)
		temp = temp.transpose()
		temp *= np.conjugate(temp)
		result[i,:,:,:] = temp.real
		
	
	freqs = np.fft.fftfreq(nperseg, 1/framerate)
	
	return result, freqs

def high_pass_filter(section, framerate, f):
	nyq_freq = framerate / 2
	numtaps = 65
	
	coeffs = signal.firwin(numtaps, np.array(f)/nyq_freq, pass_zero=False)
	return signal.lfilter(coeffs, 1, section, axis=0)
	

def read_analog_data(filename):
	infile = open(filename)
	
	signalx = np.array([])
	signaly = np.array([])
	
	while True:
		vals = np.fromfile(infile, dtype='>i4', count=2)
		if len(vals) < 2: break
		n1, n2 = vals
		block = np.fromfile(infile, dtype='>f8', count=n2)
		signalx = np.append(signalx, block)
		block = np.fromfile(infile, dtype='>f8', count=n2)
		signaly = np.append(signaly, block)
	
	infile.close()
	return np.array([signalx, signaly])


### GUI related classes ###

class MeasurementData(QObject):
	folderLoaded = pyqtSignal()

	def __init__(self, parent=None):
		QObject.__init__(self, parent)
		
		self.data_folder = None
		self.video_path = None
		self.analog_signals = None
		self.config = None
		
	@pyqtProperty(str, notify=folderLoaded)
	def folder(self):
		return self.data_folder
	
	@folder.setter
	def folder(self, folder):
		folder = folder.strip('file://')
		self.data_folder = folder
		configpath = folder + '/config.txt'
		self.config = parse_config(configpath)
		self.video_path = self.data_folder + '/Camera_Data.sav'
		analog_data_path = self.data_folder + '/Analog_Data.sav'
		nframes = get_video_nframes(self.video_path)
		self.config.nframes = nframes
		self.analog_signals = read_analog_data(analog_data_path)
		self.folderLoaded.emit()
		
	@pyqtProperty(bool, notify=folderLoaded)
	def ready(self):
		return self.config != None
	
	@pyqtProperty(int, notify=folderLoaded)
	def nFrames(self):
		return self.config.nframes
	
	@pyqtProperty(int, notify=folderLoaded)
	def framerate(self):
		return self.config.framerate
	
	@pyqtProperty(int, notify=folderLoaded)
	def image_width(self):
		return self.config.image_width
	
	@pyqtProperty(int, notify=folderLoaded)
	def image_height(self):
		return self.config.image_height
	
	def getVideoSnapshot(self, frameindex):
		config = self.config
		imgdata = read_frame(self.video_path, config.image_width, config.image_height, frameindex)
		img = QImage(imgdata, config.image_width, config.image_height, QImage.Format_Grayscale8)
		return img


class FourierAnalyzer(QObject):
	
	analysisComplete = pyqtSignal()
	
	def __init__(self, parent=None):
		QObject.__init__(self, parent)
		
		self._data = None
		self.analysis = None
		self.t0 = None
		self.x0 = None
		self.y0 = None
		self.frequencies = None
		self.max_value = None
		self.analysis_window = 50
		self.analysis_step = 30
	
	@pyqtProperty(MeasurementData)
	def measurementData(self):
		return self._data
	@measurementData.setter
	def measurementData(self, val):
		self._data = val
	
	@pyqtSlot(int, int, int, int, int, int)
	def analyze(self, x, y, t, width, height, duration):
		img_width = self._data.image_width
		img_height = self._data.image_height
		
		xstart = x
		ystart = y
		tstart = t
		xwindow = width
		ywindow = height
		twindow = duration
		
		print('Analyzing...')
		
		print('Reading input data')
		section = read_section(self._data.video_path, xstart, ystart, tstart, xwindow, ywindow, twindow, img_width, img_height)
		analysis, f = spectrogram(section, self.analysis_window, self.analysis_step, self._data.framerate)
		
		
		self.analysisComplete.emit()
		print('Done.')
		print ('Min freq:', f[1])
		
		self.analysis = analysis
		self.t0 = tstart
		self.x0 = xstart
		self.y0 = ystart
		self.frequencies = f[1:]
		self.max_value = self.analysis.max()

class BandPassAnalyzer(QObject):
	
	analysisComplete = pyqtSignal()
	
	def __init__(self, parent=None):
		QObject.__init__(self, parent)
		
		self._data = None
		self.analysis = None
		self.t0 = None
		self.x0 = None
		self.y0 = None
		self._cutoff = 0
	
	@pyqtProperty(float)
	def cutoff(self):
		return self._cutoff
	@cutoff.setter
	def cutoff(self, val):
		self._cutoff = val
		
	@pyqtProperty(MeasurementData)
	def measurementData(self):
		return self._data
	@measurementData.setter
	def measurementData(self, val):
		self._data = val
		
	@pyqtSlot(int, int, int, int, int, int)
	def analyze(self, x, y, t, width, height, duration):
		img_width = self._data.image_width
		img_height = self._data.image_height
		framerate = self._data.framerate
		nyq_freq = framerate/2
		
		black_treshold = 40
		uint8_data = read_section(self._data.video_path, x, y, t, width, height, duration, img_width, img_height)
		zeros = np.where(uint8_data < black_treshold)
		
		numtaps, beta = signal.kaiserord(30, self.cutoff/2/nyq_freq)
		coeffs = signal.firwin(numtaps, 5/nyq_freq, window=('kaiser', beta))
		baseline = signal.lfilter(coeffs, 1.0, uint8_data, axis=0)
		
		section = 100 * (uint8_data / baseline - 1)
		
		del baseline
		del uint8_data
		
		numtaps = 65
		freqs = [self._cutoff, self._cutoff*2]
		coeffs = signal.firwin(numtaps, freqs, nyq=nyq_freq, pass_zero=False)
		analysis = signal.lfilter(coeffs, 1, section, axis=0)
		
		#analysis = high_pass_filter(section, framerate, [self._cutoff, self._cutoff*2])**2
		analysis[zeros] = 0
		
		self.analysisComplete.emit()
		
		self.analysis = analysis
		self.t0 = t
		self.x0 = x
		self.y0 = y
		

class SnapshotView(QQuickPaintedItem):
	
	changed = pyqtSignal()
	
	def __init__(self, parent=None):
		QQuickPaintedItem.__init__(self, parent)
		self._data = None
		self._frame = 0

	@pyqtProperty(MeasurementData, notify=changed)
	def measurementData(self):
		return self._data
	@measurementData.setter
	def measurementData(self, val):
		self._data = val

	@pyqtProperty(int)
	def frame(self):
		return self._frame
	@frame.setter
	def frame(self, val):
		self._frame = val
		self.update()
		
	def paint(self, painter):
		if self._data is None or not self._data.ready: return
		img = self._data.getVideoSnapshot(self._frame)
		painter.drawImage(0, 0, img)
		

class AnalysisVisualization(QQuickPaintedItem):
	def __init__(self, parent=None):
		QQuickPaintedItem.__init__(self, parent)
		self._analyzer = None
		
	@pyqtProperty('QVariant')
	def analyzer(self):
		return self._analyzer
	@analyzer.setter
	def analyzer(self, val):
		self._analyzer = val
		if val:
			self._analyzer.analysisComplete.connect(self.update)

class OverlayImage(AnalysisVisualization):
	def __init__(self, parent=None):
		AnalysisVisualization.__init__(self, parent)
		self._frame = 0
		self._overlay_treshold = 0
		self._overlay_frequency = 0

	@pyqtProperty(int)
	def frame(self):
		return self._frame
	@frame.setter
	def frame(self, val):
		self._frame = val
		self.update()
		
	@pyqtProperty(int)
	def treshold(self):
		return self._overlay_treshold
	@treshold.setter
	def treshold(self, val):
		self._overlay_treshold = val
		self.update()
		
	@pyqtProperty(int)
	def frequency(self):
		return self._overlay_frequency
	@frequency.setter
	def frequency(self, val):
		self._overlay_frequency = val
		self.update()

	def paint(self, painter):
		if self.analyzer is None or self.analyzer.analysis is None: return
	
		step = (self.frame - self.analyzer.t0) // self.analyzer.analysis_step
		if step < 0 or step >= self.analyzer.analysis.shape[0]:
			return
		
		data = self.analyzer.analysis[step, self.frequency, :, :] > self.treshold

		imagearr = np.zeros((data.shape[0], data.shape[1], 4), dtype="uint8")
		imagearr[:,:, 3] = data*255
		imagearr[:,:, 0] = 255
		imagestr = imagearr.flatten().tobytes()
		image_height, image_width = data.shape
		qimg = QImage(imagestr, image_width, image_height, QImage.Format_ARGB32)
		painter.drawImage(0, 0, qimg)

class BPFOverlayImage(AnalysisVisualization):
	def __init__(self, parent=None):
		AnalysisVisualization.__init__(self, parent)
		self._frame = 0
		self._overlay_treshold = 0

	@pyqtProperty(int)
	def frame(self):
		return self._frame
	@frame.setter
	def frame(self, val):
		self._frame = val
		self.update()
		
	@pyqtProperty(float)
	def treshold(self):
		return self._overlay_treshold
	@treshold.setter
	def treshold(self, val):
		self._overlay_treshold = val
		self.update()
		
	def paint(self, painter):
		if self.analyzer is None or self.analyzer.analysis is None: return
	
		t = self.frame - self.analyzer.t0
		if t < 0 or t >= self.analyzer.analysis.shape[0]: return 
	
		data = self.analyzer.analysis[t, :, :] > self.treshold

		imagearr = np.zeros((data.shape[0], data.shape[1], 4), dtype="uint8")
		imagearr[:,:, 3] = data*255
		imagearr[:,:, 0] = 255
		imagestr = imagearr.flatten().tobytes()
		image_height, image_width = data.shape
		qimg = QImage(imagestr, image_width, image_height, QImage.Format_ARGB32)
		painter.drawImage(0, 0, qimg)


class SpectrumImage(AnalysisVisualization):
	def __init__(self, parent=None):
		AnalysisVisualization.__init__(self, parent)
		self._x = 0
		self._y = 0
		self._radius = 3
		self._cutoff = 1
		
	@pyqtProperty(int)
	def targetX(self):
		return self._x
	@targetX.setter
	def targetX(self, val):
		self._x = val
		self.update()
	
	@pyqtProperty(int)
	def targetY(self):
		return self._y
	@targetY.setter
	def targetY(self, val):
		self._y = val
		self.update()
	
	@pyqtProperty(int)
	def radius(self):
		return self._radius
	@radius.setter
	def radius(self, va):
		self._radius = val
		self.update()
	
	@pyqtProperty(float)
	def valueCutoff(self):
		return self._cutoff
	@valueCutoff.setter
	def valueCutoff(self, val):
		self._cutoff = val
		self.update()
	
	def paint(self, painter):
		if self.analyzer is None or self.analyzer.analysis is None: return
	
		analysis = self.analyzer.analysis
		xindex, yindex = self.targetX, self.targetY
		
		if xindex >= analysis.shape[-1]:
			print('x out of range:' + str(xindex))
			xindex = 0
			
		if yindex >= analysis.shape[-2]:
			print('y out of range:' + str(yindex))
			yindex = 0
		
		radius = self._radius
		yind0 = max(yindex-radius,0)
		yind1 = min(yindex+radius, analysis.shape[-2])
		xind0 = max(xindex-radius,0)
		xind1 = min(xindex+radius, analysis.shape[-1])
		data = analysis[:,1:,yind0:yind1,xind0:xind1]
		data = data.mean(axis=-1).mean(axis=-1)
		data = data.transpose()
		
		max_value = self.analyzer.max_value*self._cutoff**2
		normalized = 256 * (data/max_value)**0.5
		
		# Lines should be 32 bit aligned
		imagearr = np.zeros((normalized.shape[0], (normalized.shape[1]+3)//4 * 4))
		imagearr[:,0:normalized.shape[1]] = normalized
		
		imagestr = np.uint8(imagearr.flatten()).tobytes()
		image_height, image_width = data.shape
		qimg = QImage(imagestr, image_width, image_height, QImage.Format_Indexed8)
		ct = create_colortable()
		qimg.setColorTable(ct)
		qimg = qimg.mirrored().scaled(self.width(), self.height(), transformMode=Qt.SmoothTransformation)
		painter.drawImage(0, 0, qimg)


class AnalogSignalPlot(QQuickPaintedItem):
	def __init__(self, parent=None):
		QQuickPaintedItem.__init__(self, parent)
		self._data = None
	
	@pyqtProperty(MeasurementData)
	def measurementData(self):
		return self._data
	@measurementData.setter
	def measurementData(self, val):
		self._data = val
	
	def paint(self, painter):
		if self._data is None: return
		signals = self._data.analog_signals
		width = self.width()
		height = self.height()
		
		duration = self._data.nFrames/self._data.framerate
		nsamples = int(duration*self._data.config.analog_samplerate)
		
		y = np.abs(signals[0,:nsamples])
		y -= np.min(y)
		y /= np.max(y[100:])
		y = 1 - .9*y
		y *= height
		
		x = np.round(np.linspace(0, width, len(y)))
		
		for i in range(y.shape[0]-1):
			painter.drawLine(x[i], y[i], x[i+1], y[i+1])


def create_colortable():
	table = []
	for i in range(256):
		h = 255 - i
		s = 255
		l = i #64 + i*(256-64) // 256
		table.append( QColor.fromHsl(h, s, l).rgb() )
	return table
		


class VideoExporter(QObject):
	
	@pyqtSlot(BandPassAnalyzer, str, int)
	def saveVideoFrames(self, analyzer, folder, skip):
		snapshotView = SnapshotView()
		overlay = BPFOverlayImage()
		analogSignalPlot = AnalogSignalPlot()
		data = analyzer.measurementData
		snapshotView.measurementData = data
		overlay.analyzer = analyzer
		overlay.treshold = 1
		analogSignalPlot.measurementData = data
		
		t0 = analyzer.t0
		nframes = analyzer.analysis.shape[0] // skip
		
		width = data.image_width
		snapshot_height = data.image_height
		overlay_width = analyzer.analysis.shape[2]
		overlay_height = analyzer.analysis.shape[1]
		overlay_x = analyzer.x0
		overlay_y = analyzer.y0
		plot_height = 200
		
		analogSignalPlot.setWidth(width)
		analogSignalPlot.setHeight(plot_height)
		
		height = snapshot_height + plot_height
		
		for i in range(nframes):
			frame = t0 + i*skip
			overlay.frame = frame
			overlay.frame = frame
			analogSignalPlot.frame = frame
			
			snapshot_img = data.getVideoSnapshot(frame)
			
			overlay_img = QImage(overlay_width, overlay_height, QImage.Format_ARGB32)
			overlay_img.fill(0)
			overlay_painter = QPainter(overlay_img)
			overlay.paint(overlay_painter)
			overlay_painter.end()
			
			plot_img = QImage(width, plot_height, QImage.Format_ARGB32)
			plot_img.fill(0xffffffff)
			plot_painter = QPainter(plot_img)
			analogSignalPlot.paint(plot_painter)
			plot_painter.end()
			
			image = QImage(width, height, QImage.Format_ARGB32)
			painter = QPainter(image)
			
			painter.drawImage(0, 0, snapshot_img)
			painter.drawImage(overlay_x, overlay_y, overlay_img)
			painter.drawImage(0, snapshot_height, plot_img)
			painter.end()
			
			print('saving frame', i)
			image.save('{0}/frame{1:03d}.png'.format(folder, i))
		
		

if __name__ == '__main__':
	from PyQt5.QtQml import QQmlApplicationEngine
	from PyQt5.QtGui import QGuiApplication
	import sys
	
	app = QApplication(sys.argv)
	
	qmlRegisterType(MeasurementData, "org.dynalyzer", 1, 0, "MeasurementData");
	qmlRegisterType(BandPassAnalyzer, "org.dynalyzer", 1, 0, "BandPassAnalyzer");
	qmlRegisterType(FourierAnalyzer, "org.dynalyzer", 1, 0, "FourierAnalyzer");
	qmlRegisterType(SnapshotView, "org.dynalyzer", 1, 0, "SnapshotView");
	qmlRegisterType(OverlayImage, "org.dynalyzer", 1, 0, "OverlayImage");
	qmlRegisterType(BPFOverlayImage, "org.dynalyzer", 1, 0, "BPFOverlayImage");
	qmlRegisterType(SpectrumImage, "org.dynalyzer", 1, 0, "SpectrumImage");
	qmlRegisterType(AnalogSignalPlot, "org.dynalyzer", 1, 0, "AnalogSignalPlot");
	qmlRegisterType(VideoExporter, "org.dynalyzer", 1, 0, "VideoExporter");
		
	engine = QQmlApplicationEngine()
	engine.load('dynalyzer.qml')
	
	app.exec_()