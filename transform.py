from sklearn import base
import numpy as np
import pywt
from scipy.stats import linregress
import datapoint
from scipy.signal import decimate


class Extractor(base.BaseEstimator):
    """ Extracts features from each datapoint. """

    def __init__(self, extractor=None):
        if extractor is not None:
            self.extractor = extractor

    def transform(self, X):
        return np.array([self(x) for x in X], ndmin=2)

    def fit(self, X, y):
        return self

    def __call__(self, x):
        return self.extractor(x)


class MeanSubtract(Extractor):
    """ Subtracts the mean of the data from every point. """

    def extractor(self, x):
        m = Mean()(x)
        return [xx-m for xx in x]


class Clip(Extractor):
    """ Cut some amount from the end of the data. """

    def __init__(self, size):
        self.size = size

    def extractor(self, x):
        return x[0:int(len(x)*self.size)]


class Concat(Extractor):
    """ Reshape multi-dimensional data into one dimension. """

    def extractor(self, x):
        return np.ravel(np.array(x), 'F')


class Split(Extractor):
    """ Split data into equal sized parts. """

    def __init__(self, steps=None, divs=None):
        self.steps = steps
        self.divs = divs

    def extractor(self, x):
        steps = self.steps or len(x) / self.divs
        return np.array([x[i:i+steps] for i in range(0, len(x), steps)]).T


class Transpose(Extractor):
    """ Transpose the data. """

    def extractor(self, x):
        return x.T


class Decimate(Extractor):
    """ Shrink signal by applying a low-pass filter. """

    def __init__(self, factor):
        self.factor = factor

    def extractor(self, x):
        if self.factor == 1.0:
            return x
        return decimate(x, self.factor, ftype='fir')


class Window(Extractor):
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


class DecimateWindow(Extractor):
    """ Decimate the data at different scales and apply a function to each. """

    def __init__(self, f):
        self.f = f

    def extractor(self, x):
        results = []

        for scale in [2**e for e in range(0, 9)]:
            decimated = Decimate(scale)(x)
            results.append(self.f(decimated))

        return np.concatenate(results)


class Map(Extractor):
    """ Apply a function to each part of a feature (e.g. each electrode channel). """

    def __init__(self, f, steps=None, divs=None):
        self.f = f
        self.steps = steps
        self.divs = divs

    def extractor(self, x):
        steps = self.steps or len(x) / self.divs
        return np.ravel([self.f(x[i:i+steps]) for i in range(0, len(x), steps)])


class CrossCorrelation(Extractor):
    """ Calculate cross correlation between two signals. """

    def extractor(self, x):
        # split in two
        x1 = x[:len(x)/2]
        x2 = x[len(x)/2:]
        return np.correlate(x1, x2, 'full')


class TimeDelay(Extractor):
    """ Calculate time delay between two equal-length signals. """

    def extractor(self, x):
        cc = CrossCorrelation()(x)  # get cross correlation
        window = np.hanning(len(cc))
        cc *= window  # apply window
        return [float(cc.argmax() - (len(cc) / 2))]  # find maximum index


class Fourier(Extractor):
    """ Perform a Fourier transform on the data. """

    def extractor(self, x):
        return np.fft.rfft(x)


class DiscreteWavelet(Extractor):
    """ Perform a wavelet transform on the data. """

    def __init__(self, kind, L, D, concat=False):
        self.kind = kind
        self.L = L
        self.D = D
        self.concat = concat

    def extractor(self, x):
        wavelet = pywt.wavedec(x, self.kind, level=self.L)
        wavelet = wavelet[0:self.L-self.D]
        if self.concat:
            return np.concatenate(wavelet)
        else:
            return wavelet


class Detrend(Extractor):
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


class PostStimulus(Extractor):
    """ Remove any pre-stimulus data from the datapoint. """

    def __init__(self, offset=0):
        self.offset = offset

    def extractor(self, x):
        return x[self.offset-datapoint.window_offset:]


class PreStimulus(Extractor):
    """
    Removes any post-stimulus data.
    If the classifier can handle this, it must be infering information from
    the experiment context itself rathern than from the stimulus.
    """

    def extractor(self, x):
        return x[0:-datapoint.window_offset]


class ElectrodeAvg(Extractor):
    """ Take the average of the two electrode values. """

    def extractor(self, x):
        return [(xx[0] + xx[1]) / 2.0 for xx in x]


class ElectrodeDiff(Extractor):
    """ Take the difference of the two electrode values. """

    def extractor(self, x):
        return [xx[0] - xx[1] for xx in x]


class MovingAvg(Extractor):
    """ Take a moving average of time series data. """

    def __init__(self, n):
        self.n = n

    def extractor(self, x):
        mov_avg = []

        # maintains a running sum of window
        accum = sum(x[:self.n])

        for i, xx in enumerate(x):
            # calculate mean from accumulator
            avg = accum / self.n
            mov_avg.append(avg)

            # add end of window to accumulator
            try:
                accum += x[i+self.n]
            except IndexError:
                # leave loop when reach end
                break

            # remove start of window from accumulator
            accum -= x[i]

        return mov_avg


class Noise(Extractor):
    """ Extract noise from data. """

    def __init__(self, n):
        self.mov_avg = MovingAvg(n)
        self.n = n

    def extractor(self, x):
        smoothed = self.mov_avg(x)
        return [xx - ss for xx, ss in zip(x[self.n/2:-self.n/2], smoothed)]


class FeatureEnsemble(Extractor):
    """ Take an ensemble of different features from the data. """

    def extractor(self, x):
        avg = Mean(x)
        diff1 = Mean()(Abs()(Differential()(x)))
        diff2 = Mean()(Abs()(Differential()(Differential()(x))))

        vari = Var()(x)
        vardiff1 = Var()(Differential()(x))
        vardiff2 = Var()(Differential()(Differential()(x)))

        hjorth_mob = vardiff1**0.5 / vari**0.5
        hjorth_com = (vardiff2**0.5 / vardiff1**0.5) / hjorth_mob

        skew = Skewness()(x)
        kurt = Kurtosis()(x)

        return [avg, diff1, diff2, vari, vardiff1, vardiff2,
                hjorth_mob, hjorth_com, skew, kurt]


class Abs(Extractor):
    """ Return absolute values. """

    def extractor(self, x):
        return map(abs, x)


class Differential(Extractor):
    """ The change in x. """

    def extractor(self, x):
        return [x2 - x1 for (x1, x2) in zip(x[:-1], x[1:])]


class Mean(Extractor):
    """ The average of x. """

    def extractor(self, x):
        return sum(x) / len(x)


class Moment(Extractor):
    """ The nth central moment. """

    def __init__(self, n):
        self.n = n

    def extractor(x, n):
        m = Mean()(x)
        return sum([(xx-m)**n for xx in x]) / len(x)


class Var(Moment):
    """ The variance of x. """

    def __init__(self):
        Var.__init__(self, 2)


class Stdev(Extractor):
    """ The standard deviation of x. """

    def extractor(self, x):
        return Var()(x)**0.5


class Skewness(Extractor):
    """ The sample skewness of x. """

    def extractor(self, x):
        return Moment(3)(x) / (Var()(x) ** (3/2))


class Kurtosis(Extractor):
    """ The sample kurtosis of x. """

    def extractor(self, x):
        return Moment(4)(x) / (Var()(x) ** 2) - 3
