from sklearn import base, svm, lda, qda, pipeline, preprocessing, grid_search
import pywt
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import decimate
from scipy.stats import linregress
from itertools import chain, groupby

import plant
import datapoint


labels = ['null', 'ozone', 'H2SO4']


class FeatureExtractor(base.BaseEstimator):
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
        if self.factor == 1.0:
            return x
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
                window *= np.hanning(len(window))
            windows.append(self.f(window))

        return np.concatenate(windows)


class DecimateWindowTransform(FeatureExtractor):
    """ Decimate the data at different scales and apply a function to each. """

    def __init__(self, f):
        self.f = f

    def extractor(self, x):
        results = []
        
        for scale in [2**e for e in range(0, 9)]:
            decimated = DecimateTransform(scale).extractor(x)
            results.append(self.f(decimated))

        return np.concatenate(results)


class MapElectrodeTransform(FeatureExtractor):
    """ Apply a function to each electrode. """

    def __init__(self, f):
        self.f = f

    def extractor(self, x):
        return np.array(zip(*[self.f(np.array(xx)) for xx in zip(*x)]))


class DiscreteWaveletTransform(FeatureExtractor):
    """ Perform a wavelet transform on the data. """

    def __init__(self, kind, L, D):
        self.kind = kind
        self.L = L
        self.D = D

    def extractor(self, x):
        wavelet = pywt.wavedec(x, self.kind, self.L)
        return np.concatenate(wavelet[0:self.L-self.D])


class DetrendTransform(FeatureExtractor):
    """ Remove any linear trends in the data. """

    def linear(self, xs, m, c):
        return map(lambda xx: m*xx + c, xs)

    def extractor(self, x):
        # find best fitting line to pre-stimulus window
        times = range(0, len(x))
        m, c, r, p, err = linregress(times[0:-datapoint.window_offset], 
                                     x[0:-datapoint.window_offset])
        # subtract extrapolated line from data to produce new dataset
        return x - self.linear(times, m, c)

class PostStimulusTransform(FeatureExtractor):
    """ Remove any pre-stimulus data from the datapoint. """

    def __init__(self, offset=0):
        self.offset = offset

    def extractor(self, x):
        return x[self.offset-datapoint.window_offset:]


class ElectrodeAvgTransform(FeatureExtractor):
    """ Take the average of the two electrode values. """

    def extractor(self, x):
        return [(xx[0] + xx[1]) / 2.0 for xx in x]


class ElectrodeDiffTransform(FeatureExtractor):
    """ Take the difference of the two electrode values. """

    def extractor(self, x):
        return [xx[0] - xx[1] for xx in x]


class MovingAvgTransform(FeatureExtractor):
    """ Take a moving average of time series data. """

    def __init__(self, n):
        self.n = n

    def extractor(self, x):
        mov_avg = []

        for i, xx in enumerate(x):
            start = i-self.n/2
            if start < 0:
                start = 0
            end = i+self.n/2
            if end > len(x):
                end = len(x)
            window = x[start:end]
            mov_avg.append(mean(window))

        return mov_avg


class FeatureEnsembleTransform(FeatureExtractor):
    """ Take an ensemble of different features from the data. """

    def extractor(self, x):
        diff = mean(map(abs, differential(x)))
        noise = mean(map(abs, differential(differential(x))))
        vari = var(x)
        vardiff = var(differential(x))
        varnoise = var(differential(differential(x)))
        hjorth_mob = vardiff**0.5 / vari**0.5
        hjorth_com = (varnoise**0.5 / vardiff**0.5) / hjorth_mob
        return [diff, noise, vari, vardiff, hjorth_mob, hjorth_com]


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
    X, y = datapoint.generate_all(plants)
    # filter to relevant datapoint types
    X, y = datapoint.filter_types(X, y, labels)
    # balance the dataset
    X, y = datapoint.balance(X, y, False)
    
    # take the average and detrend the data ahead of time
    X = ElectrodeAvgTransform().transform(X)
    X = DetrendTransform().transform(X)
    X = PostStimulusTransform(60).transform(X)

    return X, y


def plot_features(f1, f2):
    # load plant data from files
    plants = plant.load_all()
    # preprocess data
    X, y = preprocess(plants)

    # scale data
    X = FeatureEnsembleTransform().transform(X)
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    groups = lambda: datapoint.group_types(X, y)

    # visualize the feature extractor
    colors = iter(cm.rainbow(np.linspace(0, 1, len(list(groups())))))
    for dtype, (Xg, yg) in groups():
        plt.scatter(Xg[:,f1], Xg[:,f2], c=next(colors), label=dtype)
    plt.legend()
    plt.show()


def plot_histogram(feature):
    # load plant data from files
    plants = plant.load_all()
    # preprocess data
    X, y = preprocess(plants)

    groups = lambda: datapoint.group_types(X, y)

    # visualize a histogram of the feature
    for dtype, (Xg, yg) in groups():
        Xg = FeatureEnsembleTransform().transform(Xg)
        plt.hist(Xg[:,feature], bins=40, alpha=0.5, label=dtype)
    plt.legend()
    plt.show()


_ensemble = FeatureEnsembleTransform().extractor
_window = WindowTransform(_ensemble, 3, False).extractor
pre_pipe = [
    ('feature', DecimateWindowTransform(_window)),
    ('scaler', preprocessing.StandardScaler())
]


def plot_pipeline():
    # load plant data from files
    plants = plant.load_all()
    # preprocess data
    X, y = preprocess(plants)

    # transform data on pipeline
    lda_ = lda.LDA(2)
    lda_pipe = pipeline.Pipeline(pre_pipe + [('lda', lda_)])
    lda_pipe.fit(X, y)
    yp = lda_pipe.predict(X)
    X = lda_pipe.transform(X)

    groups = lambda: datapoint.group_types(X, y)

    # visualize the pipeline 
    cgen = lambda: iter(cm.rainbow(np.linspace(0, 1, len(list(groups())))))
    colors = cgen()
    for dtype, (Xg, yg) in groups():
        tp = (y == yp)[yp==dtype]
        Xtp, Xfp = X[tp], X[~tp]
        c = next(colors)
        plt.scatter(Xtp[:,0], Xtp[:,1], 'o', c=c, label=dtype)
        plt.scatter(Xfp[:,0], Xfp[:,1], '.', c=c, label=dtype + " false positive")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # load plant data from files
    plants = plant.load_all()

    # split plant data into training and validation sets
    random.shuffle(plants)
    train_len = int(0.75 * len(plants))
    train_plants = plants[:train_len]
    valid_plants = plants[train_len:]

    print "Experiments in training set:", len(train_plants)
    print "Experiments in validation set:", len(valid_plants)

    # get X data and y labels
    X_train, y_train = preprocess(train_plants)
    X_valid, y_valid = preprocess(valid_plants)

    print "Datapoints in training set:", len(X_train)
    class_train = [(d[0], len(list(d[1]))) for d in groupby(y_train)]
    print "Classes in training set:", class_train 
    print "Datapoints in validation set:", len(X_valid)
    class_valid = [(d[0], len(list(d[1]))) for d in groupby(y_valid)]
    print "Classes in validation set:", class_valid

    # set up pipeline
    pipeline = pipeline.Pipeline(pre_pipe + [('svm', svm.SVC())])
    params = [{}]

    # perform grid search on pipeline, get best parameters from training data
    grid = grid_search.GridSearchCV(pipeline, params, cv=5, verbose=2)
    grid.fit(X_train, y_train)
    classifier = grid.best_estimator_

    print "Grid search results:"
    print grid.best_score_

    # test the classifier on the validation data set
    validation_score = classifier.fit(X_train, y_train).score(X_valid, y_valid)

    print "Validation data results:"
    print validation_score
