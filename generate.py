import plant
import datapoint
import plot
import transform
import duet

from sklearn import pipeline, decomposition, preprocessing
import numpy as np
import scipy
import pywt
import matplotlib.pyplot as plt

# load all plant data in directory
print "Loading data"
plants = plant.load_all()

# process all data
print "Processing data"
X, y, sources = datapoint.generate_all(plants)

# write data to file
print "Writing to data.csv"
datapoint.save("data.csv", X, y, sources)

concat = transform.Concat()
split = transform.Split(divs=2)
detrend = transform.Map(transform.Detrend(), divs=2)
transpose = transform.Transpose()


def plot_plants():
    plot.plant_data_save(plants, "plants")


def plot_datapoints():
    plot.datapoints_save(X, y, "datapoints")


def plot_detrended():
    # plot detrended points
    pipe = pipeline.Pipeline([('c', concat), ('d', detrend), ('s', split)])
    plot.datapoints_save(pipe.transform(X), y, 'detrend')


def plot_wavelets():
    # plot wavelets
    wavelets = transform.DiscreteWavelet('haar', 11, 0)
    avg = transform.ElectrodeAvg()
    plot.datapoints_save(wavelets.transform(avg.transform(X)), y,
                         'wavelet', plot.datapoint_set)


mov_avg = transform.Map(transform.MovingAvg(100), divs=2)
deriv = transform.Map(transform.Differential(), divs=2)
mean = transform.Map(transform.MeanSubtract(), divs=2)


def plot_derivatives():
    # plot first derivatives
    pipe = pipeline.Pipeline(
        [('c', concat), ('m', mov_avg), ('d', deriv), ('me', mean), ('s', split)])
    plot.datapoints_save(pipe.transform(X), y, 'deriv')

abs_ = transform.Abs()


def plot_derivatives_abs():
    # plot first derivatives absolute value
    pipe = pipeline.Pipeline(
        [('c', concat), ('m', mov_avg), ('d', deriv),
         ('me', mean), ('a', abs_), ('s', split)])
    plot.datapoints_save(pipe.transform(X), y, 'deriv_abs')


def plot_derivatives_diff():
    # plot first derivatives differences
    pipe = pipeline.Pipeline(
        [('c', concat), ('m', mov_avg), ('d', deriv),
         ('me', mean), ('a', abs_), ('s', split),
         ('e', transform.ElectrodeDiff())])
    plot.datapoints_save(pipe.transform(X), y, 'deriv_diff')

correl = transform.CrossCorrelation()
window = transform.Extractor(lambda x: x * np.hanning(len(x)))


def plot_cross_correlation():
    # plot cross correlation of electrode channels
    pipe = pipeline.Pipeline([('c', concat), ('m', mov_avg), ('d', deriv),
                              ('me', mean), ('cr', correl), ('w', window)])
    plot_func = lambda xx, yy: plot.datapoint(xx, yy, False)
    plot.datapoints_save(pipe.transform(X), y, 'correlation', plot_func)


def plot_cross_correlation_abs():
    # plot cross correlation of electrode channels with absolute valued data
    pipe = pipeline.Pipeline([('c', concat), ('m', mov_avg), ('d', deriv), ('me', mean),
                              ('a', abs_), ('cr', correl), ('w', window)])
    plot_func = lambda xx, yy: plot.datapoint(xx, yy, False)
    plot.datapoints_save(pipe.transform(X), y, 'correlation_abs', plot_func)


def plot_fourier():
    # plot fourier transform
    pipe = pipeline.Pipeline(
        [('c', concat),
         ('p', transform.Map(transform.PostStimulus(), divs=2)),
         ('w', transform.Map(window, divs=2)),
         ('f', transform.Map(transform.Fourier(), divs=2)),
         ('s', split)])

    def plot_func(xx, yy):
        plt.ylim(0.0, 0.1)
        plot.datapoint(xx, yy, False)
    plot.datapoints_save(pipe.transform(X), y, 'fourier', plot_func)


noise = transform.Map(transform.Noise(1000), divs=2)


def plot_noise():
    pipe = pipeline.Pipeline([('c', concat), ('n', noise),
                              ('s', split)])
    plot.datapoints_save(pipe.transform(X), y, 'noise')


def plot_noise_correlation():
    pipe = pipeline.Pipeline(
        [('c', concat), ('n', noise), ('m', mov_avg), ('d', deriv), ('me', mean),
         ('a', abs_), ('cr', correl)])
    plot_func = lambda xx, yy: plot.datapoint(xx, yy, False)
    plot.datapoints_save(pipe.transform(X), y, 'noise_correlation', plot_func)


def plot_ica():
    ica = decomposition.FastICA(max_iter=1000)
    T = [ica.fit_transform(x) for x in X]
    plot.datapoints_save(T, y, 'ica')


def plot_ica_plants():
    ica = decomposition.FastICA(max_iter=1000)
    R = [p.readings for p in plants]
    T = [ica.fit_transform(r) for r in R]
    n_plants = [plant.PlantData(p.name, r, p.stimuli, p.sample_freq)
                for p, r in zip(plants, T)]
    plot.plant_data_save(n_plants, 'ica_plants')


def plot_ica_wobble():
    # wobble readings back and forth to try and deal with time delay!

    class Wobble(transform.Extractor):
        def __init__(self, k):
            self.k = k

        def transform(self, X):
            print X.shape
            x1 = X[:, 0]
            x2 = X[:, 1]
            print x1.shape
            print x2.shape

            wob_left = [x1[self.k-i:-self.k-i].T for i in range(self.k)]
            wob_right = [x2[self.k+i:-self.k+i].T for i in range(self.k)]
            wobbles = np.append(wob_left, wob_right, 0)
            print wobbles.shape
            return wobbles.T

    ica = decomposition.FastICA(2)
    R = [p.readings for p in plants]
    pipe = pipeline.Pipeline([('w', Wobble(30)), ('i', ica)])
    T = [pipe.fit_transform(r) for r in R]
    n_plants = [plant.PlantData(p.name, r, p.stimuli, p.sample_freq)
                for p, r in zip(plants, T)]
    plot.plant_data_save(n_plants, 'ica_wobble')


def plot_ica_duet():
    # plot ICA using DUET algorithm
    du = duet.DUET(numsources=4)
    R = [p.readings for p in plants]
    T = [du.transform(r) for r in R]
    n_plants = [plant.PlantData(p.name, r, p.stimuli, p.sample_freq)
                for p, r in zip(plants, T)]
    plot.plant_data_save(n_plants, 'ica_duet')


def plot_ica_noise():
    # plot ICA on low-level noise
    ica = decomposition.FastICA(max_iter=1000)
    T = [ica.fit_transform(split.extractor(noise.extractor(concat.extractor(x))))
         for x in X]
    plot.datapoints_save(T, y, 'ica_noise')


def plot_ica_duet_noise():
    # plot ICA using DUET algorithm
    du = duet.DUET(numsources=4)
    T = [du.transform(split.extractor(noise.extractor(concat.extractor(x))))
         for x in X]
    plot.datapoints_save(T, y, 'ica_duet_noise')


def plot_mult_noise():
    # multiply the noise between the two channels
    # this should identify correlated and anti-correlated behaviour
    mult = transform.ElectrodeOp(lambda x1, x2: x1 * x2)
    pipe = pipeline.Pipeline([('n', noise), ('m', mult)])
    plot.datapoints_save(pipe.transform(X), y, 'mult_noise')


def plot_ica_wavelets():
    # plot ICA performed between electrode channels on each wavelet level

    def wave_ica(x):
        ica = decomposition.FastICA(max_iter=10000)
        # calculate wavelet transform of each
        w1 = pywt.wavedec(x[:, 0], 'haar', level=12)
        w2 = pywt.wavedec(x[:, 1], 'haar', level=12)
        # calculate ica between components
        t = [ica.fit_transform(np.array([lev1, lev2]).T)
             for lev1, lev2 in zip(w1, w2)]
        return np.array(t)

    extractor = transform.Extractor(wave_ica)
    T = extractor.transform(X)
    plot.datapoints_save(T, y, 'wavelet_ica', plot.datapoint_set)


def plot_power_spectral_density():
    # plot power spectral density of data
    pipe = pipeline.Pipeline(
        [('c', concat),
         ('p', transform.Map(transform.PostStimulus(), divs=2)),
         ('w', transform.Map(transform.PowerSpectralDensityAvg(4096), divs=2)),
         ('s', split)])

    def plot_func(xx, yy):
        plot.datapoint(xx, yy, False)
    plot.datapoints_save(pipe.transform(X), y, 'psd', plot_func)


def plot_power_spectral_density_2d():
    # plot power spectral density of data in 2D
    pipe = pipeline.Pipeline(
        [('a', transform.ElectrodeAvg()),
         ('p', transform.PostStimulus()),
         ('w', transform.PowerSpectralDensity(256))])

    def plot_func(xx, yy):
        plt.pcolormesh(xx.T)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
    plot.datapoints_save(pipe.transform(X), y, 'psd2d', plot_func)
