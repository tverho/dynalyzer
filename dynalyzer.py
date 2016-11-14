import numpy as np
import os
from matplotlib import pyplot
import xml.etree.ElementTree as ET
from PyQt5.QtCore import QObject, pyqtSlot, pyqtProperty, pyqtSignal, Qt, QUrl
from PyQt5.QtGui import QImage, QTransform, QColor, QPainter
from PyQt5.QtWidgets import QApplication
from PyQt5.QtQuick import QQuickImageProvider, QQuickPaintedItem
from PyQt5.QtQml import qmlRegisterType
from scipy import signal, ndimage

class VideoRawData:
	def __init__(self, folder, config, rotate = True):
		filenames = get_video_filenames(folder)
		w, h = config.image_width, config.image_height
		
		self.arrays = [self.create_view(f, w, h) for f in filenames]
		total_nframes = sum(a.shape[0] for a in self.arrays)
		if rotate:
			self.shape = (total_nframes, w, h)
		else:
			self.shape = (total_nframes, h, w)
		self.rotate = rotate
		
	def create_view(self, filename, width, height):
		header_size = 4
		m = np.memmap(filename, dtype='u1', offset=header_size)
		framesize = width*height
		filesize = len(m) + header_size
		nframes = filesize // (framesize+header_size)
		
		# Each 8 byte sequence is in reversed order, so create a 4d
		# array and reverse the last axis to get the bytes in order
		shape = (nframes, height, width//8, 8)
		
		# Use stride tricks to skip the header bytes between frames
		strides = ((framesize+header_size), width, 8, 1)
		strided = np.lib.stride_tricks.as_strided(m, strides=strides, shape=shape)
		view = strided[:,:,:,::-1]
		return view
	
	
	def __getitem__(self, slices):
		clean_slices = []
		unpack_indices = [slice(None), slice(None), slice(None)]
		try: list(slices)
		except: slices = list((slices,))
		
		for i in range(3):
			try:
				slices[i].start
				clean_slices.append(slices[i])
			except IndexError:
				clean_slices.append(slice(None))
			except AttributeError:
				# This is an integer index
				clean_slices.append(slice(slices[i], slices[i]+1))
				unpack_indices[i] = 0
		
		array = self.get_section(*clean_slices)
		return array[unpack_indices]
		
	def get_section(self, tslice, yslice, xslice):
		start, end, step = tslice.indices(self.shape[0])
		if self.rotate:
			start1, end1, step1 = xslice.indices(self.shape[2])
			start1, end1, step1 = -start1-1, -end1-1, -step1
			start2, end2, step2 = yslice.indices(self.shape[1])
		else:
			start1, end1, step1 = yslice.indices(self.shape[1])
			start2, end2, step2 = xslice.indices(self.shape[2])
		
		rect = (slice(None), slice(start1, end1, step1), slice(start2 // 8, end2 // 8))
		section = None
		for arr in self.arrays:
			if section is not None:
				if end > arr.shape[0]:
					section = np.vstack((section, arr[::step][rect]))
				else:
					section = np.vstack((section, arr[:end:step][rect]))
					break
			
			elif end <= arr.shape[0]:
				section = arr[start:end:step][rect]
				break
			
			elif start < arr.shape[0]:
				section = arr[start::step][rect]
					
			start -= arr.shape[0]
			end -= arr.shape[0]
		
		section = section.reshape((section.shape[0], section.shape[1], section.shape[2]*section.shape[3]))
		if self.rotate:
			section = section.transpose((0, 2, 1))
		
		return section


def parse_config(filename, rotate=True):
	tree = ET.parse(filename)
	root = tree.getroot()

	class Object(): pass
	config = Object()
	
	width = int( root.find(".//*[Name='Image Width']/Val").text )
	height = int( root.find(".//*[Name='Image Height']/Val").text )
	config.image_width = width
	config.image_height = height
	#nframes = int( root.find(".//*[Name='Frames to read']/Val").text )
	config.framerate = float( root.find(".//*[Name='Frames per second']/Val").text )
	config.analog_samplerate = float( root.find(".//*[Name='Sample Rate (S/s)']/Val").text )
	
	return config

def get_video_filenames(folder):
	files = os.listdir(folder)
	result = [os.path.join(folder, file) for file in files if file.count('Camera_Data') > 0]
	result.sort()
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
		self.video_data = None
		self.analog_signals = None
		self.config = None
		
	@pyqtProperty(str, notify=folderLoaded)
	def folder(self):
		return self.data_folder
	
	@folder.setter
	def folder(self, folder):
		folder = QUrl(folder).toLocalFile()
		self.data_folder = folder
		configpath = os.path.join(folder, 'config.txt')
		self.config = parse_config(configpath)
		self.video_data = VideoRawData(folder, self.config)
		analog_data_path = os.path.join(self.data_folder, 'Analog_Data.sav')
		self.analog_signals = read_analog_data(analog_data_path)
		self.folderLoaded.emit()
		
	@pyqtProperty(bool, notify=folderLoaded)
	def ready(self):
		return self.video_data is not None
	
	@pyqtProperty(int, notify=folderLoaded)
	def nFrames(self):
		return self.video_data.shape[0]
	
	@pyqtProperty(int, notify=folderLoaded)
	def framerate(self):
		return self.config.framerate
	
	@pyqtProperty(int, notify=folderLoaded)
	def image_width(self):
		return self.video_data.shape[2]
	
	@pyqtProperty(int, notify=folderLoaded)
	def image_height(self):
		return self.video_data.shape[1]
	
	def getVideoSnapshot(self, frameindex):
		imgdata = self.video_data[frameindex]
		qimg = to_aligned_qimage(imgdata, QImage.Format_Grayscale8)
		return qimg


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
		self.analysis_window = 200
		self.analysis_step = 100
	
	@pyqtProperty(MeasurementData)
	def measurementData(self):
		return self._data
	@measurementData.setter
	def measurementData(self, val):
		self._data = val
	
	@pyqtSlot(int, int, int, int, int, int)
	def analyze(self, x, y, t, width, height, duration):
		
		
		print('Analyzing...')
		print('Reading input data')
		section = self._data.video_data[t:t+duration, y:y+height, x:x+width]
		print('Calculating')
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
		self._lower_limit = 0
		self._upper_limit = None
		self._remove_baseline = True
		self._temporal_averaging = 10
		self.downsampling = 1
	
	@pyqtProperty(float)
	def lowerLimit(self):
		return self._lower_limit
	@lowerLimit.setter
	def lowerLimit(self, val):
		self._lower_limit = val
	
	@pyqtProperty(float)
	def upperLimit(self):
		return self._upper_limit
	@upperLimit.setter
	def upperLimit(self, val):
		self._upper_limit = val
	
	@pyqtProperty(MeasurementData)
	def measurementData(self):
		return self._data
	@measurementData.setter
	def measurementData(self, val):
		self._data = val
		
	@pyqtSlot(int, int, int, int, int, int)
	def analyze(self, x, y, t, width, height, duration):
		del self.analysis
		
		print('Analyzing...')
		
		framerate = self._data.framerate
		nyq_freq = framerate / 2
		
		if False:
			framerate = self._data.framerate
			min_nyq_freq = self._upper_limit
			nyq_freq = framerate / 2
			self.downsampling = int(nyq_freq / min_nyq_freq)
			framerate /= self.downsampling
			nyq_freq = framerate / 2
	
		print('Reading input data')
		uint8_data = self._data.video_data[t:t+duration:self.downsampling, y:y+height, x:x+width]
		
		black_treshold = 40
		zeros = np.where(uint8_data < black_treshold)
		
		if self._remove_baseline:
			print('Calculating baseline')
			# Running mean
			f = self._lower_limit / 2
			N = int(framerate / f)
			baseline = np.zeros(uint8_data.shape, dtype='f4')
			cumsum = np.cumsum(uint8_data, axis=0)
			baseline[N:,:,:] = (cumsum[N:,:,:] - cumsum[:-N,:,:]) / N
			baseline[:N,:,:] = np.mean(uint8_data[:N,:,:], axis=0)
			del cumsum
		else:
			baseline = np.mean(uint8_data)
		
		section = np.zeros(uint8_data.shape, dtype='f4')
		section[:,:,:] = 100 * (uint8_data/baseline - 1)
		del baseline
		del uint8_data
		
		print('Filtering')
		numtaps = 65
		freqs = [self._lower_limit, self._upper_limit]
		if freqs[1] > nyq_freq:
			freqs = [freqs[0]]
		coeffs = signal.firwin(numtaps, freqs, nyq=nyq_freq, pass_zero=False)
		analysis = signal.lfilter(coeffs, 1, section, axis=0)
		
		analysis[zeros] = 0
		
		del section
		analysis = np.abs(analysis)
		if self._temporal_averaging:
			#N = self._temporal_averaging
			N = framerate / (self._lower_limit + self._upper_limit)
			b = np.ones(N) / N
			analysis = signal.lfilter(b, 1, analysis, axis=0)
					
		self.analysis = analysis
		self.t0 = t
		self.x0 = x
		self.y0 = y
		self.analysisComplete.emit()
		print('Done.')
		

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

		imagearr = np.zeros((data.shape[0], data.shape[1], 4), dtype="u1")
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
		self._smooth_radius = 1.5

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
	
		t = (self.frame - self.analyzer.t0) // self.analyzer.downsampling
		if t < 0 or t >= self.analyzer.analysis.shape[0]: return 
	
		frame = np.abs(self.analyzer.analysis[t, :, :])
		if self._smooth_radius:
			frame = ndimage.gaussian_filter(frame, self._smooth_radius)
	
		imagearr = np.zeros((frame.shape[0], frame.shape[1], 4), dtype="uint8")

		for i in range(5):
			treshold = self.treshold * (1 + i*0.5)
			pixels = np.where(frame > treshold)
			imagearr[:,:,0][pixels] = (255 - i*40)
			if i==0: 
				imagearr[:,:,3][pixels] = 255

		imagestr = imagearr.flatten().tobytes()
		image_height, image_width = frame.shape
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
		
		qimg = to_aligned_qimage(normalized, QImage.Format_Indexed8)
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

		y = signals[0,:nsamples]
		y -= np.min(y)

		y /= np.max(y[len(y)//4:])
		y = 1 - y
		y *= height
		
		x = np.linspace(0, width, len(y))
		
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
		

def to_aligned_qimage(imgarr, format):
	aligned = np.zeros((imgarr.shape[0], (imgarr.shape[1]+3)//4 * 4), dtype='uint8')
	aligned[:,0:imgarr.shape[1]] = imgarr
		
	image_height, image_width = imgarr.shape
	qimg = QImage(aligned.tobytes(), image_width, image_height, format)
	return qimg


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
		
	if len(sys.argv) > 1:
		path = QUrl(sys.argv[1])
		engine.rootContext().setContextProperty("loadpath", path)
	
	engine.load('dynalyzer.qml')
	app.exec_()

