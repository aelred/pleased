from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import random

import plant
import datapoint


class FeatureEnsemble(BaseEstimator):

	def transform(self, X):
		Xt = []
		for x in X:
			x = elec_avg(x)
			noise = mean(map(abs, differential(differential(x))))
			variance = var(x)
			Xt.append([noise, variance])
		return Xt

	def fit(self, X, y):
		return self


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

def preprocess(plants):
	# extract windows from plant data
	datapoints = datapoint.generate_all(plants)
	# balance the dataset
	datapoints = datapoint.balance(datapoints)

	# remove any pre-stimulus data
	datapoints = map(datapoint.post_stimulus, datapoints)

	# extract features and labels
	labels = [d[0] for d in datapoints]
	data = [d[1] for d in datapoints]
	return data, labels

if __name__ == "__main__":
	# load plant data from files
	plants = plant.load_all()

	# split plant data into training and validation sets
	random.shuffle(plants)
	train_len = int(0.75 * len(plants))
	train_plants = plants[:train_len]
	valid_plants = plants[train_len:]

	# get X data and y labels
	X_train, y_train = preprocess(train_plants)
	X_valid, y_valid = preprocess(valid_plants)

	# set up pipeline
	extractor = FeatureEnsemble()
	classifier = SVC(kernel='linear')
	pipeline = Pipeline([('extractor', extractor), ('classifier', classifier)])

	# perform 5-fold cross validation on pipeline
	score = cross_val_score(pipeline, X_train, y_train, cv=5)
	print score

	# TODO: Validate on train/validate sets