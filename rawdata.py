import numpy as np
import xml.etree.ElementTree as ET
import os

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
	config.force_scale = float( root.find(".//*[Name='Force Scale (N/v)']/Val").text )
	config.displacement_scale = float( root.find(".//*[Name='Strain Scale (mm/V)']/Val").text )
	
	return config


def get_video_filenames(folder):
	files = os.listdir(folder)
	result = [os.path.join(folder, file) for file in files if file.count('Camera_Data') > 0]
	result.sort()
	return result


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
