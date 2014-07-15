from sklearn import base
import numpy as np
from scipy.stats import linregress
import datapoint
from scipy.signal import decimate

class Extractor(base.BaseEstimator):
    """ Extracts features from each datapoint. """

    def __init__(self, extractor=None):
        if extractor is not None:
            self.extractor = extractor

    def transform(self, X):
        return np.array([self.extractor(x) for x in X], ndmin=2)

    def fit(self, X, y):
        return self


class MeanSubtractTransform(Extractor):
    """ Subtracts the mean of the data from every point. """

    def extractor(self, x):
        m = mean(x)
        return [xx-m for xx in x]


class ClipTransform(Extractor):
    """ Cut some amount from the end of the data. """

    def __init__(self, size):
        self.size = size

    def extractor(self, x):
        return x[0:int(len(x)*self.size)]


class DecimateTransform(Extractor):
    """ Shrink signal by applying a low-pass filter. """

    def __init__(self, factor):
        self.factor = factor

    def extractor(self, x):
        if self.factor == 1.0:
            return x
        return decimate(x, self.factor, ftype='fir')


class WindowTransform(Extractor):
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


class DecimateWindowTransform(Extractor):
    """ Decimate the data at different scales and apply a function to each. """

    def __init__(self, f):
        self.f = f

    def extractor(self, x):
        results = []
        
        for scale in [2**e for e in range(0, 9)]:
            decimated = DecimateTransform(scale).extractor(x)
            results.append(self.f(decimated))

        return np.concatenate(results)


class MapElectrodeTransform(Extractor):
    """ Apply a function to each electrode. """

    def __init__(self, f):
        self.f = f

    def extractor(self, x):
        return np.array(zip(*[self.f(np.array(xx)) for xx in zip(*x)]))


class DiscreteWaveletTransform(Extractor):
    """ Perform a wavelet transform on the data. """

    def __init__(self, kind, L, D):
        self.kind = kind
        self.L = L
        self.D = D

    def extractor(self, x):
        wavelet = pywt.wavedec(x, self.kind, self.L)
        return np.concatenate(wavelet[0:self.L-self.D])


class DetrendTransform(Extractor):
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

class PostStimulusTransform(Extractor):
    """ Remove any pre-stimulus data from the datapoint. """

    def __init__(self, offset=0):
        self.offset = offset

    def extractor(self, x):
        return x[self.offset-datapoint.window_offset:]


class PreStimulusTransform(Extractor):
    """ 
    Removes any post-stimulus data.
    If the classifier can handle this, it must be infering information from
    the experiment context itself rathern than from the stimulus.
    """

    def extractor(self, x):
        return x[0:-datapoint.window_offset]


class ElectrodeAvgTransform(Extractor):
    """ Take the average of the two electrode values. """

    def extractor(self, x):
        return [(xx[0] + xx[1]) / 2.0 for xx in x]


class ElectrodeDiffTransform(Extractor):
    """ Take the difference of the two electrode values. """

    def extractor(self, x):
        return [xx[0] - xx[1] for xx in x]


class MovingAvgTransform(Extractor):
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


class FeatureEnsembleTransform(Extractor):
    """ Take an ensemble of different features from the data. """

    def extractor(self, x):
        avg = mean(x)
        diff1 = mean(map(abs, differential(x)))
        diff2 = mean(map(abs, differential(differential(x))))

        vari = var(x)
        vardiff1 = var(differential(x))
        vardiff2 = var(differential(differential(x)))

        hjorth_mob = vardiff1**0.5 / vari**0.5
        hjorth_com = (vardiff2**0.5 / vardiff1**0.5) / hjorth_mob

        skew = skewness(x)
        kurt = kurtosis(x)

        return [avg, diff1, diff2, vari, vardiff1, vardiff2, 
                hjorth_mob, hjorth_com, skew, kurt]


def differential(x):
    """
    Returns: The change in x.
    """
    return [x2 - x1 for (x1, x2) in zip(x[:-1], x[1:])]


def mean(x):
    """ Returns: The average of x. """
    return sum(x) / len(x)


def moment(x, n):
    """ Returns: The nth central moment. """
    m = mean(x)
    return sum([(xx-m)**n for xx in x]) / len(x)


def var(x):
    """ Returns: The variance of x. """
    return moment(x, 2)


def stdev(x):
    """ Returns: The standard deviation of x. """
    return var(x)**0.5


def skewness(x):
    """ Returns: The sample skewness of x. """
    return moment(x, 3) / (var(x) ** (3/2))


def kurtosis(x):
    """ Returns: The sample kurtosis of x. """
    return moment(x, 4) / (var(x) ** 2) - 3