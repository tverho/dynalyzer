import numpy as np
from matplotlib import pyplot
import xml.etree.ElementTree as ET
from PyQt5.QtCore import QObject, pyqtSlot, pyqtProperty, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QTransform
from PyQt5.QtQuick import QQuickImageProvider
from PyQt5.QtQml import qmlRegisterType

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

def analyze_section(section, window, step):
	nframes, ny, nx = section.shape
	ntransforms = (nframes-window)//step
	window += window%2
	
	result = np.zeros([ntransforms, window//2 + 1, ny, nx], dtype="float32")
	
	# shape: (x, y, t)
	transposed = np.array(section.transpose())
	
	for i in range(ntransforms):
		temp = np.fft.rfft(transposed[:,:, i*step:i*step+window])
		result[i,:,:,:] = np.abs(temp.transpose())
	
	return result 


def read_analog_data(filename):
	infile = open(filename)
	
	result = np.array([])
	
	while True:
		vals = np.fromfile(infile, dtype='>i4', count=2)
		if len(vals) < 2: break
		n1, n2 = vals
		section = np.fromfile(infile, dtype='>f8', count=n1*n2)
		result = np.append(result, section)
	
	infile.close()
	return result.reshape([-1, 2])

### GUI related classes


class SpectrumImageProvider(QQuickImageProvider):
	def __init__(self, analyzer):
		QQuickImageProvider.__init__(self, QQuickImageProvider.Image)
		
		self.analyzer = analyzer
		
	
	def requestImage(self, id, requested_size):
		x_str, y_str = id.split(',');
		
		xindex = int(x_str)
		yindex = int(y_str)
				
		if xindex >= self.analyzer.analysis.shape[-1]:
			print('x out of range:' + str(xindex))
			xindex = 0
			
		if yindex >= self.analyzer.analysis.shape[-2]:
			print('y out of range:' + str(yindex))
			yindex = 0
			
		data = self.analyzer.analysis[:,1:,yindex,xindex].transpose()
		
		normalized = 256*(data/self.analyzer.max_value)**0.3
		
		# Lines should be 32 bit aligned
		imagearr = np.zeros((normalized.shape[0], (normalized.shape[1]+3)//4 * 4))
		imagearr[:,0:normalized.shape[1]] = normalized
		
		imagestr = np.uint8(imagearr.flatten()).tobytes()
		image_height, image_width = data.shape
		qimg = QImage(imagestr, image_width, image_height, QImage.Format_Grayscale8)
		qimg = qimg.mirrored().scaled(800, 600)
		
		self.image = qimg # avoid gc
		return qimg, qimg.size()
	

class OverlayImageProvider(QQuickImageProvider):
	def __init__(self, analyzer):
		QQuickImageProvider.__init__(self, QQuickImageProvider.Image)
		
		self.analyzer = analyzer
	
	
	def requestImage(self, id, requested_size):
		pos_str, frequency_str, treshold_str = id.split(',')
		
		frameindex = int(pos_str)
		frequency = int(frequency_str)
		treshold = float(treshold_str)
		
		step = (frameindex - self.analyzer.t0) // self.analyzer.analysis_step
		
		data = self.analyzer.analysis[step, frequency, :, :] > treshold

		imagearr = np.zeros((data.shape[0], data.shape[1], 4), dtype="uint8")
		imagearr[:,:, 3] = data*255
		imagearr[:,:, 0] = 255
		imagestr = imagearr.flatten().tobytes()
		image_height, image_width = data.shape
		qimg = QImage(imagestr, image_width, image_height, QImage.Format_ARGB32)
		
		self.image = qimg
	
		return qimg, qimg.size()
			
	
class FourierAnalyzer(QObject):
	
	folderLoaded = pyqtSignal()
	
	def __init__(self, folder, parent=None):
		QObject.__init__(self, parent)
		
		self.data_folder = None
		self.video_path = None
		self.analog_signals = None
		self.analysis = None
		self.config = None
		self.t0 = None
		self.x0 = None
		self.y0 = None
		self.max_value = None
		self.analysis_window = 50
		self.analysis_step = 5
		
		self.folder = folder
	
	@pyqtProperty(str, notify=folderLoaded)
	def folder(self):
		return self.data_folder
	
	@folder.setter
	def folder(self, folder):
		folder = folder.strip('file:')
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
	
	@pyqtSlot(int, int, int, int, int, int)
	def analyze(self, x, y, t, width, height, duration):
		config = self.config
		img_width = config.image_width
		img_height = config.image_height
		
		# rotated 90
		xstart = x
		ystart = y
		tstart = t
		xwindow = width
		ywindow = height
		twindow = duration
		
		print('Analyzing...')
		
		section = read_section(self.video_path, xstart, ystart, tstart, xwindow, ywindow, twindow, img_width, img_height)
		analysis = analyze_section(section, self.analysis_window, self.analysis_step)
		
		print('Done.')
		
		self.analysis = analysis
		self.t0 = tstart
		self.x0 = xstart
		self.y0 = ystart
		self.max_value = analysis.max()
	
	def getVideoSnapshot(self, frameindex):
		config = self.config
		imgdata = read_frame(self.video_path, config.image_width, config.image_height, frameindex)
		img = QImage(imgdata, config.image_width, config.image_height, QImage.Format_Grayscale8)
		return img


class SnapshotProvider(QQuickImageProvider):
	def __init__(self, analyzer):
		QQuickImageProvider.__init__(self, QQuickImageProvider.Image)
		self.analyzer = analyzer
	
	def requestImage(self, id, requested_size):
		if analyzer.config == None: return
		frameindex = int(id)
		img = analyzer.getVideoSnapshot(frameindex)
		if requested_size.isValid():
			img = img.scaled(requested_size), self.image.size()
		
		self.image = img # avoid GC
		return img, img.size()


if __name__ == '__main__':
	from PyQt5.QtQml import QQmlApplicationEngine
	from PyQt5.QtGui import QGuiApplication
	import sys
	
	app = QApplication(sys.argv)
	
	#analyzer = FourierAnalyzer()
	
	qmlRegisterType(FourierAnalyzer, "com.dynalyzer", 1, 0, "FourierAnalyzer");
	
	snapshotProvider = SnapshotProvider(analyzer)
	spectrumProvider = SpectrumImageProvider(analyzer)
	overlayProvider = OverlayImageProvider(analyzer)
		
	engine = QQmlApplicationEngine()
	#engine.rootContext().setContextProperty('analyzer', analyzer)
	engine.addImageProvider('snapshot', snapshotProvider)
	engine.addImageProvider('analysis', spectrumProvider)
	engine.addImageProvider('overlay', overlayProvider)
	engine.load('dynalyzer.qml')
	
	app.exec_()

