from sklearn import base, decomposition
import numpy as np
import pywt
from scipy.stats import linregress
import datapoint
from scipy.signal import decimate
import parmap
import itertools


class Extractor(base.BaseEstimator):
    """ Extracts features from each datapoint. """

    def __init__(self, extractor=None):
        if extractor is not None:
            self.extractor = extractor

    def transform(self, X):
        return np.array(parmap.parmap(self.extractor, X), ndmin=2)

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


class Average2D(Extractor):
    """ Reduce size of a 2D array by averaging. """

    def __init__(self, factors):
        self.factors = factors

    def extractor(self, x):
        new = x.reshape([x.shape[0] / self.factors[0], self.factors[0],
                         x.shape[1] / self.factors[1], self.factors[1]])
        return new.mean(-1).mean(1)


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


class Histogram(Extractor):
    """ Take a histogram of data. """

    def __init__(self, num_bins):
        self.num_bins = num_bins

    def fit(self, X, y):
        # calculate mean and standard deviation of dataset
        x = np.concatenate([xx for xx in X])
        stdev = np.std(x)
        mean = np.mean(x)
        # set range to one standard deviation
        self.range = (mean-stdev, mean+stdev)
        return self

    def extractor(self, x):
        return np.histogram(x, self.num_bins, self.range)[0]


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
    """
    Apply a function to each part of a feature (e.g. each electrode channel).
    The function can either be a single function or a list of functions.
    """

    def __init__(self, f, steps=None, divs=None):
        try:
            iter(f)
        except TypeError:
            self.fs = itertools.repeat(f)
        else:
            self.fs = f

        self.steps = steps
        self.divs = divs

    def fit(self, X, y):
        print X.shape
        steps = self.steps or X.shape[1] / self.divs
        try:
            for i, f in zip(range(0, X.shape[1], steps), self.fs):
                f.fit(X[:, i:i+steps], y)
        except AttributeError:
            # f is a function, not a transformer
            pass
        return self

    def extractor(self, x):
        steps = self.steps or len(x) / self.divs
        return np.ravel([f(x[i:i+steps]) for i, f in
                         zip(range(0, len(x), steps), self.fs)])


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
        # find maximum index in first half (assume delayed in one direction)
        return [float(len(cc) / 2) - cc[:len(cc) / 2 + 1].argmax()]


class Fourier(Extractor):
    """ Perform a Fourier transform on the data. """

    def extractor(self, x):
        return np.fft.rfft(x)


class PowerSpectralDensity(Extractor):
    """ Calculate power spectral density of time series in 2D. """

    def __init__(self, window_size, window_step=None):
        self.window_size = window_size
        self.window_step = window_step or window_size / 2

    def extractor(self, x):
        # take short-time windowed fourier transforms of data
        psd = []
        for i in range(0, len(x)-self.window_size, self.window_step):
            # get window
            w = np.hanning(self.window_size) * x[i:i+self.window_size]
            # calculate periodogram of window
            fft = np.fft.rfft(w)
            fw = np.log(np.sqrt(np.square(np.real(fft)) + np.square(np.imag(fft))))
            # add to power spectral density
            psd.append(fw)

        return np.array(psd)


class PowerSpectralDensityAvg(PowerSpectralDensity):
    """ Calculate power spectral density of time series averaged. """

    def extractor(self, x):
        psd2d = PowerSpectralDensity.extractor(self, x)
        return psd2d.sum(axis=0)


class DiscreteWavelet(Extractor):
    """ Perform a wavelet transform on the data. """

    def __init__(self, kind, L, D, concat=False, transforms=None):
        self.kind = kind
        self.L = L
        self.D = D
        self.concat = concat
        self.transforms = transforms

    def fit(self, X, y):
        if self.transforms is None:
            return self

        # take wavelets
        concat = self.concat
        self.concat = False
        transforms = list(self.transforms)
        self.transforms = None
        wavelets = self.transform(X)
        self.concat = concat
        # run fit function over every wavelet level
        self.transforms = [t.fit(w.T, y) for t, w in zip(transforms, wavelets.T)]
        return self

    def extractor(self, x):
        wavelet = pywt.wavedec(x, self.kind, level=self.L)
        wavelet = wavelet[self.D:]

        # transform every wavelet level
        if self.transforms is not None:
            wavelet = [t(w) for t, w in zip(self.transforms, wavelet)]

        if self.concat:
            return np.concatenate(wavelet)
        else:
            return np.array(wavelet)


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


class ElectrodeOp(Extractor):
    """ Perform some operation between the two electrode channels. """

    def __init__(self, op=None):
        if op is not None:
            self.op = op

    def extractor(self, x):
        try:
            return [self.op(xx[0], xx[1]) for xx in x]
        except IndexError:
            # if data is concatenated
            x = x.reshape((-1, 2))
            return self.extractor(x)


class ElectrodeAvg(ElectrodeOp):
    """ Take the average of the two electrode values. """

    def op(self, x1, x2):
        return (x1 + x2) / 2


class ElectrodeDiff(ElectrodeOp):
    """ Take the difference of the two electrode values. """

    def op(self, x1, x2):
        return x1 - x2


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


class ICA(Extractor):
    """ Perform fast ICA over every element. """

    def __init__(self):
        self.ica = decomposition.FastICA()

    def extractor(self, x):
        return np.ravel(self.ica.fit_transform(x.reshape(-1, 2)), 'F')


class FeatureEnsemble(Extractor):
    """ Take an ensemble of different features from the data. """

    def extractor(self, x):
        m = Mean()
        v = Var()
        a = Abs()
        d = Differential()

        avg = m(x)
        diff1 = m(a(d(x)))
        diff2 = m(a(d(d(x))))

        vari = v(x)
        vardiff1 = v(d(x))
        vardiff2 = v(d(d(x)))

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

    def extractor(self, x):
        m = Mean()(x)
        return sum([(xx-m)**self.n for xx in x]) / len(x)


class Var(Moment):
    """ The variance of x. """

    def __init__(self):
        Moment.__init__(self, 2)


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
