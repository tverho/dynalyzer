import numpy as np
import os
from rawdata import VideoRawData, parse_config, read_analog_data
from PyQt5.QtCore import QObject, pyqtSlot, pyqtProperty, pyqtSignal, Qt, QUrl
from PyQt5.QtGui import QImage, QTransform, QColor, QPainter
from PyQt5.QtQuick import QQuickImageProvider, QQuickPaintedItem
from PyQt5.QtQml import qmlRegisterType
from scipy import signal, ndimage
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import matplotlib.cm


def map_property(type, attrname, notify=None, on_modified=None):
	def getter(self):
		return getattr(self, attrname)
	def setter(self, val):
		setattr(self, attrname, val)
		if notify: getattr(self, notify).emit()
		if on_modified: getattr(self, on_modified)()
	if notify:
		return pyqtProperty(type, getter, setter, notify)
	else:
		return pyqtProperty(type, getter, setter)


def tidy_analog_signal(y):
	sorted_y = np.sort(y)
	
	# If data is mostly negative, switch sign
	seventy_percent = int(len(y)*.8)
	if sorted_y[seventy_percent] < 0:
		y = -y
		sorted_y = np.sort(y)
	
	# Let 1% of data be negative
	one_percent = int(len(y)*.01)
	y = y - sorted_y[one_percent]
		
	return y


def plot_analog_signals(data, width, height):
	signals = data.analog_signals
	
	duration = data.nFrames/data.framerate
	nsamples = int(duration*data.analogSamplerate)

	y = signals[0,:nsamples]
	y = tidy_analog_signal(y)
	
	q = int(len(y) / width)
	if q > 1:
		# Remove excess data points
		y = y[::q]
	
	t = np.arange(0, len(y)) / data.analogSamplerate
	
	pyplot.clf()
	fig = pyplot.gcf()
	dpi = fig.get_dpi()
	fig.set_size_inches(width / dpi, height / dpi)
	
	pyplot.plot(t, y, color='black')
	
	first_second = int(1*data.analogSamplerate/q)
	pyplot.ylim(ymin=0, ymax=np.max(y[first_second:]))
	pyplot.xlim(xmin=t[0], xmax=t[-1])
	return fig


def to_aligned_qimage(imgarr, format):
	aligned = np.zeros((imgarr.shape[0], (imgarr.shape[1]+3)//4 * 4), dtype='uint8')
	aligned[:,0:imgarr.shape[1]] = imgarr
		
	image_height, image_width = imgarr.shape
	qimg = QImage(aligned.tobytes(), image_width, image_height, format)
	return qimg


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
		if self.video_data:
			return self.video_data.shape[0]
		else:
			return 0
	
	@pyqtProperty(int, notify=folderLoaded)
	def framerate(self):
		return self.config.framerate
	
	@pyqtProperty(int, notify=folderLoaded)
	def analogSamplerate(self):
		return self.config.analog_samplerate
	
	@pyqtProperty(int, notify=folderLoaded)
	def image_width(self):
		return self.video_data.shape[2]
	
	@pyqtProperty(int, notify=folderLoaded)
	def image_height(self):
		return self.video_data.shape[1]
	
	def getVideoSnapshot(self, frameindex, contrast=None, brightness=None):
		imgdata = self.video_data[frameindex]
		
		if contrast != None and brightness != None:
			maxvalue = (255 - brightness)/contrast
			saturated_pixels = np.where(imgdata > maxvalue)
			imgdata = imgdata*contrast + brightness
			imgdata[saturated_pixels] = 255
		
		image_height, image_width = imgdata.shape
		data = np.zeros((image_height, image_width, 4), dtype='u1')
		data[:,:,0] = imgdata
		data[:,:,1] = imgdata
		data[:,:,2] = imgdata
		data[:,:,3] = 255
		
		self.bytes = data.tobytes()
		qimg = QImage(self.bytes, image_width, image_height, QImage.Format_ARGB32)
		#qimg = to_aligned_qimage(imgdata, QImage.Format_Grayscale8)
		return qimg


class DifferenceAnalyzer(QObject):
    
	parametersChanged = pyqtSignal()
    
	def __init__(self, parent=None):
		QObject.__init__(self, parent)
		
		self._data = None
		self._interval = 10
		self._temporal_averaging = None
		self._black_treshold = 20
		self._relative_mode = True
		self._fast_averaging = True
    
	measurementData = map_property(MeasurementData, '_data', notify='parametersChanged')
	interval = map_property(int, '_interval', notify='parametersChanged')
	temporalAveraging = map_property(int, '_temporal_averaging', notify='parametersChanged')
	blackTreshold = map_property(int, '_black_treshold', notify='parametersChanged')
	relativeMode = map_property(bool, '_relative_mode', notify='parametersChanged')
	fastTemporalAveraging = map_property(bool, '_fast_averaging', notify='parametersChanged') 

	def analyzeSnapshot(self, t):
		t_averaging = self._temporal_averaging
		t0 = t - self._interval
		interval = self._interval
		video_data = self._data.video_data
		
		if t_averaging:
			if t0-t_averaging < 0: return None
			if self._fast_averaging:
				cur = np.mean(video_data[t-t_averaging:t], axis=0)
				prev = np.mean(video_data[t0-t_averaging:t0], axis=0)
				difference = np.abs(cur - prev)
			else:
				datablock = np.array(video_data[t0-t_averaging:t], dtype=int)
				cur = datablock[-t_averaging:]
				prev = datablock[-interval-t_averaging:-interval]
				difference = np.mean(np.abs(cur-prev), axis=0)
			
		else:
			cur = np.array(video_data[t], dtype=int)
			prev = video_data[t0]
			difference = np.abs(cur - prev)
		
		valid = np.where(video_data[t] > self._black_treshold)
		
		result = np.zeros_like(difference, dtype=float)
		if self._relative_mode:
			result[valid] = 100 * difference[valid] / video_data[t][valid]
		else:
			result[valid] = difference[valid]
		return result



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
		

class DifferenceOverlayImage(QQuickPaintedItem):
	
	changed = pyqtSignal()
	
	def __init__(self, parent=None):
		QQuickPaintedItem.__init__(self, parent)
		self._frame = 0
		self._overlay_treshold = 1
		self._smooth_radius = 0
		self._analyzer = None
		self.colormap = matplotlib.cm.get_cmap('hsv')

	analyzer = map_property('QVariant', "_analyzer", on_modified='analyzer_added')
	frame = map_property(int, "_frame", on_modified='update')
	treshold = map_property(float, "_overlay_treshold", on_modified='update')
	spatialAveraging = map_property(float, "_smooth_radius", on_modified='update')
	interval = map_property(int, "_interval", on_modified='update')
	
	def analyzer_added(self):
		if self._analyzer: 
			self._analyzer.parametersChanged.connect(self.update)

	def paint(self, painter):
		self.draw_frame(painter, self._frame)
	
	def draw_frame(self, painter, t):
		if self.analyzer is None or not self.isVisible(): return
	
		frame = self.analyzer.analyzeSnapshot(t)
		if frame is None: return
	
		if self._smooth_radius:
			frame = ndimage.gaussian_filter(frame, self._smooth_radius)
	
		imagearr = np.zeros((frame.shape[0], frame.shape[1], 4), dtype="uint8")
		        
		pixels = np.where(frame > self.treshold)
		normalized_values = (frame[pixels]-self.treshold) / (self.treshold*5)
		colors = self.colormap(normalized_values)
		imagearr[:,:,][pixels] = 255*colors

		imagestr = imagearr.flatten().tobytes()
		image_height, image_width = frame.shape
		qimg = QImage(imagestr, image_width, image_height, QImage.Format_ARGB32)
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
		data = self._data
		width, height = self.width(), self.height()
		fig = plot_analog_signals(data, width, height)
		pyplot.xticks([])
		pyplot.yticks([])
		pyplot.subplots_adjust(left=0, top=1, right=1, bottom=0)
		fig.canvas.draw()
		img = QImage(fig.canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
		painter.drawImage(0, 0, img)
		

class Exporter(QObject):
	@pyqtSlot(MeasurementData, 'QVariant', int, str)
	def saveImage(self, data, analysis_overlay, frame, fileurl):
		url = QUrl(fileurl)
		filename = url.toLocalFile()
		img = self.get_frame_image(data, analysis_overlay, frame)
		if not img.save(filename):
			print('Error saving image.')
	
	@pyqtSlot(MeasurementData, 'QVariant', 'QVariant', str, bool, bool)
	def saveImageSeries(self, data, analysis_overlay, frames, folder_url, frameRange=True, addAnalogSignalPlot=False):
		url = QUrl(folder_url)
		folder = url.toLocalFile()
		extension = '.png'
		frames = frames.toVariant()
		if frameRange:
			frames = range(*(int(s) for s in frames))
		else:
			frames = [int(s) for s in frames]
		
		if addAnalogSignalPlot:
			width = data.image_width
			plot_height = int(data.image_height * 0.25)
			plot, plot_area = self.get_analog_signals_plot(data, width, plot_height)
			master_image = QImage(width, data.image_height + plot_height, QImage.Format_ARGB32)
			painter = QPainter(master_image)
			
		for frame in frames:
			filename = os.path.join(folder, '{:06d}{}'.format(frame, extension))
			print ('Saving', filename)
			snapshot = self.get_frame_image(data, analysis_overlay, frame)
			if addAnalogSignalPlot:
				painter.drawImage(0,0, snapshot)
				painter.drawImage(0, data.image_height, plot)
				cursor_x = plot_area[0] + frame/data.nFrames * plot_area[2]
				cursor_y1 = data.image_height + plot_area[1]
				cursor_y2 = data.image_height + plot_area[1] + plot_area[3]
				painter.setPen(QColor(0,0,255))
				painter.drawLine(cursor_x, cursor_y1, cursor_x, cursor_y2)
				img = master_image
			else:
				img = snapshot
			if not img.save(filename):
					print('Error saving image.')
		
		if addAnalogSignalPlot:
			painter.end()
			
	@pyqtSlot(MeasurementData, str)
	def saveAnalogSignals(self, data, fileurl):
		url = QUrl(fileurl)
		filename = url.toLocalFile()
		signals = data.analog_signals.T
		framerate = data.framerate
		samplerate = data.analogSamplerate
		nsamples = signals.shape[0]
		config = data.config
		
		displacement = signals[:,1] * config.displacement_scale
		load = signals[:,0] * config.force_scale
		
		outdata = np.zeros((nsamples, 4))
		outdata[:, 0] = np.arange(nsamples) / samplerate
		outdata[:, 1] = np.arange(nsamples) * (framerate/samplerate)
		outdata[:, 2] = displacement
		outdata[:, 3] = load
		
		header = 'time(s) video_frame displacement(m) load(N)'
		format = ('%1.8e', '%8d', '%1.8e', '%1.8e')
		np.savetxt(filename, outdata, fmt=format, header = header)
		
	def get_frame_image(self, data, analysis_overlay, frame):
		img = data.getVideoSnapshot(frame)
		if analysis_overlay:
			img = img.convertToFormat(QImage.Format_RGB32)
			painter = QPainter(img)
			analysis_overlay.draw_frame(painter, frame)
			painter.end()
		return img
	
	def get_analog_signals_plot(self, data, width, height):
		fig = plot_analog_signals(data, width, height)
		pyplot.xlabel('Time (s)')
		pyplot.ylabel('Load (N)')
		left, top, right, bottom = .15, .1, .1, .2
		pyplot.subplots_adjust(left=left, top=1-top, right=1-right, bottom=bottom)
		fig.canvas.draw()
		img = QImage(fig.canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
		plot_area = (left*width, top*height, (1-left-right)*width, (1-top-bottom)*height)
		return img, plot_area


if __name__ == '__main__':
	from PyQt5.QtQml import QQmlApplicationEngine
	from PyQt5.QtWidgets import QApplication
	import sys
	
	app = QApplication(sys.argv)
	
	qmlRegisterType(MeasurementData, "org.dynalyzer", 1, 0, "MeasurementData");
	qmlRegisterType(DifferenceAnalyzer, "org.dynalyzer", 1, 0, "DifferenceAnalyzer");
	qmlRegisterType(SnapshotView, "org.dynalyzer", 1, 0, "SnapshotView");
	qmlRegisterType(DifferenceOverlayImage, "org.dynalyzer", 1, 0, "DifferenceOverlayImage");
	qmlRegisterType(AnalogSignalPlot, "org.dynalyzer", 1, 0, "AnalogSignalPlot");
	qmlRegisterType(Exporter, "org.dynalyzer", 1, 0, "Exporter");
		
	engine = QQmlApplicationEngine()
		
	if len(sys.argv) > 1:
		path = QUrl(sys.argv[1])
		engine.rootContext().setContextProperty("loadpath", path)
	
	dir_path = os.path.dirname(__file__)
	engine.load(os.path.join(dir_path, 'dynalyzer.qml'))
	app.exec_()

