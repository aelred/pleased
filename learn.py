from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import plant
import datapoint


class FeatureExtractor(BaseEstimator):

	def __init__(self, extractor):
		self.extractor = extractor

	def transform(self, X):
		return np.array([self.extractor(x) for x in X], ndmin=2)

	def fit(self, X, y):
		return self


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
	datapoints = datapoint.filter_types(datapoints, ["null", "ozone", "H2SO"])
	# remove any pre-stimulus data
	datapoints = map(datapoint.post_stimulus, datapoints)

	# balance the dataset
	datapoints = datapoint.balance(datapoints)

	# extract features and labels
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
	extractor = FeatureExtractor(elec_avg)
	scaler = StandardScaler()
	classifier = LDA()
	pipeline = Pipeline([('extractor', extractor), 
						 ('scaler', scaler), 
						 ('classifier', classifier)])

	# perform 5-fold cross validation on pipeline
	score = cross_val_score(pipeline, X_train, y_train, cv=5)
	print score

	# TODO: Validate on train/validate sets