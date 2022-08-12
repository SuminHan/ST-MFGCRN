import os
import h5py
import numpy as np
import pandas as pd
from os.path import join as pjoin

#from config import DATA_PATH

class Scaler:
	def __init__(self, data, region):
		self.max = data.max(0)

		np.save(f'test/max_{region}', self.max)
	
	def transform(self, data):
		return data / self.max - 0.5
	
	def inverse_transform(self, data):
		return (data + 0.5) * self.max
		
class Scaler:
	def __init__(self, data, region):
		# self.mean = np.mean(data)
		# self.std = np.std(data)

		# np.save(f'test/meanstd_{region}', [self.mean, self.std])
		pass

	def transform(self, data):
		# return (data - self.mean) / self.std
		return data

	def inverse_transform(self, data):
		# return data * self.std + self.mean
		return data

class Scaler2:
	def __init__(self, data):
		self.mean = np.mean(data)
		self.std = np.std(data)
	
	def transform(self, data):
		return (data - self.mean) / self.std
	
	def inverse_transform(self, data):
		return data * self.std + self.mean

def load_h5(filename, keywords):
	f = h5py.File(filename, 'r')
	data = []
	for name in keywords:
		data.append(np.array(f[name]))
	f.close()
	if len(data) == 1:
		return data[0]
	return data
