from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import decimate
from scipy.optimize import curve_fit
from itertools import chain

import plant
import datapoint


labels = ['null', 'ozone', 'H2SO4']


class FeatureExtractor(BaseEstimator):
	""" Extracts features from each datapoint. """

	def __init__(self, extractor):
		self.extractor = extractor

	def transform(self, X):
		return np.array([self.extractor(x) for x in X], ndmin=2)

	def fit(self, X, y):
		return self


class FeatureWindow(BaseEstimator):
	""" Extracts features from each window in a datapoint. """

	def __init__(self, extractor):
		def extract_windows(x):
			# perform extractor on every window and then chain them together
			return np.array(chain([extractor(xx) for xx in x]))

		self.extractor = FeatureExtractor(extract_windows).transform


class MeanSubtractTransform(FeatureExtractor):
	""" Subtracts the mean of the data from every point. """

	def __init__(self):
		def mean_subtract(x):
			m = mean(x)
			return [xx-m for xx in x]
		self.extractor = mean_subtract


class ClipTransform(FeatureExtractor):
	""" Cut some amount from the end of the data. """

	def __init__(self, size):
		self.extractor = lambda x: x[0:len(x)*size]


class DecimateTransform(FeatureExtractor):
	""" Shrink signal by applying a low-pass filter. """

	def __init__(self, factor):
		self.extractor = lambda x: decimate(x, factor, ftype='fir')


class WindowTransform(FeatureExtractor):
	""" Apply a function to overlapping windows. """

	def __init__(self, f, N, hanning=True):
		def window(x):
			window_size = len(x) / N
			step = window_size / 2

			windows = []
			for i in range(0, len(x)-window_size, step):
				window = x[i:i+window_size]
				if hanning:
					window *= np.hanning
				windows.append(f(window))

			return np.concatenate(windows)

		self.extractor = window


class DetrendTransform(FeatureExtractor):
	""" Remove any linear trends in the data. """

	def __init__(self):

		def linear(x, m, c):
        	return map(lambda xx: m*xx + c, x)

		def detrend(x):
			# find best fitting curve to pre-stimulus window
	        times = range(0, len(x))
	        params, cov = curve_fit(linear, times[0:-datapoint.window_offset], 
	                                x[0:-datapoint.window_offset], (0, 0))
	        # subtract extrapolated curve from data to produce new dataset
	        return x - linear(times, *params)

	    self.extractor = detrend


class PostStimulusTransform(FeatureExtractor):
	""" Remove any pre-stimulus data from the datapoint. """

	def __init__(self, offset=0.0):
		self.extractor = lambda x: x[datapoint.window_offset-offset:]


def features(x):
	"""
	Returns: A list of features extracted from the datapoint x.
	"""
	x = elec_avg(x)
	diff = mean(map(abs, differential(x)))
	noise = mean(map(abs, differential(differential(x))))
	std = stdev(x)
	stdiff = stdev(differential(x))
	return [diff, noise, std, stdiff]


def elec_avg(x):
	"""
	Params:
		x: Data comprised of two columns from two electrodes.
	Returns: The data averaged into a single column.
	"""
	return [(xx[0] + xx[1]) / 2.0 for xx in x]


def elec_diff(x):
	"""
	Params:
		x: Data comprised of two columns from two electrodes.
	Returns: A vector giving the difference of the two columns.
	"""
	return [xx[0] - xx[1] for xx in x]


def differential(x):
	"""
	Returns: The change in x.
	"""
	return [x2 - x1 for (x1, x2) in zip(x[:-1], x[1:])]

def mean(x):
	""" Returns: The average of x. """
	return sum(x) / len(x)

def var(x):
	""" Returns: The variance of x. """
	m = mean(x)
	return sum([(xx-m)**2 for xx in x]) / len(x)

def stdev(x):
	""" Returns: The standard deviation of x. """
	return var(x)**0.5

def preprocess(plants):
	# extract windows from plant data
	datapoints = datapoint.generate_all(plants)
	# filter to relevant datapoint types
	datapoints = datapoint.filter_types(datapoints, labels)
	# remove any pre-stimulus data
	datapoints = map(datapoint.post_stimulus, datapoints)

	# balance the dataset
	datapoints = datapoint.balance(datapoints)

	return datapoints

def extract(datapoints):
	datapoints = list(datapoints)
	labels = [d[0] for d in datapoints]
	data = [d[1] for d in datapoints]
	return data, np.asarray(labels)

def plot_features():
	# load plant data from files
	plants = plant.load_all()
	# preprocess data
	datapoints = preprocess(plants)

	# scale data
	X, y = extract(datapoints)
	X = FeatureExtractor(features).transform(X)
	scaler = StandardScaler()
	scaler.fit(X)

	groups = lambda: datapoint.group_types(datapoints)

	# visualize the feature extractor
	colors = iter(cm.rainbow(np.linspace(0, 1, len(list(groups())))))
	for dtype, points in groups():
		X, y = extract(points)
		X = FeatureExtractor(features).transform(X)
		X = scaler.transform(X)
		plt.scatter(X[:,0], X[:,1], c=next(colors))
	plt.show()

if __name__ == "__main__":
	# load plant data from files
	plants = plant.load_all()

	# split plant data into training and validation sets
	random.shuffle(plants)
	train_len = int(0.75 * len(plants))
	train_plants = plants[:train_len]
	valid_plants = plants[train_len:]

	# get X data and y labels
	X_train, y_train = extract(preprocess(train_plants))
	X_valid, y_valid = extract(preprocess(valid_plants))

	# set up pipeline
	window = FeatureExtractor(lambda x: window(x, 6000, 600))
	extractor = FeatureExtractor(elec_avg)
	scaler = StandardScaler()
	classifier = QDA()
	pipeline = Pipeline([('extractor', extractor), 
						 ('scaler', scaler), 
						 ('classifier', classifier)])

	# perform 5-fold cross validation on pipeline
	score = cross_val_score(pipeline, X_train, y_train, cv=5)
	print score

	# TODO: Validate on train/validate sets