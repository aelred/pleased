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

	def __init__(self, extractor=None):
		if extractor is not None:
			self.extractor = extractor

	def transform(self, X):
		return np.array([self.extractor(x) for x in X], ndmin=2)

	def fit(self, X, y):
		return self


class MeanSubtractTransform(FeatureExtractor):
	""" Subtracts the mean of the data from every point. """

	def extractor(self, x):
		m = mean(x)
		return [xx-m for xx in x]


class ClipTransform(FeatureExtractor):
	""" Cut some amount from the end of the data. """

	def __init__(self, size):
		self.size = size

	def extractor(self, x):
		return x[0:int(len(x)*self.size)]


class DecimateTransform(FeatureExtractor):
	""" Shrink signal by applying a low-pass filter. """

	def __init__(self, factor):
		self.factor = factor

	def extractor(self, x):
		return decimate(x, self.factor, ftype='fir')


class WindowTransform(FeatureExtractor):
	""" Apply a function to overlapping windows. """

	def __init__(self, f, N, hanning=True):
		self.f = f
		self.N = N
		self.hanning = hanning

	def extractor(self, x):
		window_size = 2 * len(x) / (self.N + 1)
		step = window_size / 2

		windows = []
		for i in range(0, len(x)-window_size+1, step):
			window = x[i:i+window_size]
			if self.hanning:
				window *= np.hanning
			windows.append(self.f(window))

		return np.concatenate(windows)


class DetrendTransform(FeatureExtractor):
	""" Remove any linear trends in the data. """

	def extractor(self, x):
		def linear(x, m, c):
			return map(lambda xx: m*xx + c, x)

		# find best fitting curve to pre-stimulus window
		times = range(0, len(x))
		params, cov = curve_fit(linear, times[0:-datapoint.window_offset], 
								x[0:-datapoint.window_offset], (0, 0))
		# subtract extrapolated curve from data to produce new dataset
		return x - linear(times, *params)

class PostStimulusTransform(FeatureExtractor):
	""" Remove any pre-stimulus data from the datapoint. """

	def __init__(self, offset):
		self.offset = offset

	def extractor(self, x):
		return x[datapoint.window_offset-self.offset:]


class ElectrodeAvgTransform(FeatureExtractor):
	""" Take the average of the two electrode values. """

	def extractor(self, x):
		return [(xx[0] + xx[1]) / 2.0 for xx in x]


class ElectrodeDiffTransform(FeatureExtractor):
	""" Take the difference of the two electrode values. """

	def extractor(self, x):
		return [xx[0] - xx[1] for xx in x]


class FeatureEnsembleTransform(FeatureExtractor):
	""" Take an ensemble of different features from the data. """

	def extractor(self, x):
		diff = mean(map(abs, differential(x)))
		noise = mean(map(abs, differential(differential(x))))
		std = stdev(x)
		stdiff = stdev(differential(x))
		return [diff, noise, std, stdiff]


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
	pipeline = Pipeline([('elec_avg', ElectrodeAvgTransform()),
						 ('detrend', DetrendTransform()),
						 ('poststim', PostStimulusTransform(0)),
						 ('scaler', StandardScaler()), 
						 ('classifier', LDA())])

	# perform 5-fold cross validation on pipeline
	cross_val_score = cross_val_score(pipeline, X_train, y_train, cv=5)

	print "Cross-validation results:"
	print cross_val_score

	# fit a classifier to the whole of the training data
	classifier = pipeline.fit(X_train, y_train)

	# test the classifier on training and validation data sets
	train_score = pipeline.score(X_train, y_train)
	valid_score = pipeline.score(X_valid, y_valid)

	print "Training data results:"
	print train_score

	print "Validation data results:"
	print valid_score