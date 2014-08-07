import plant
import datapoint
import plot
import transform

from sklearn import pipeline, decomposition
import numpy as np
import scipy
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


def plot_plants():
    plot.plant_data_save(plants)


def plot_datapoints():
    plot.datapoints_save(X, y)


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
        [('c', concat), ('p', transform.PostStimulus()),
         ('w', transform.Map(window, divs=2)),
         ('f', transform.Map(transform.Fourier(), divs=2)),
         ('s', split)])

    def plot_func(xx, yy):
        plt.yscale('log')
        plot.datapoint(xx, yy, False)
    plot.datapoints_save(pipe.transform(X), y, 'fourier', plot_func)


noise = transform.Map(transform.Noise(100), divs=2)


def plot_noise():
    pipe = pipeline.Pipeline([('c', concat), ('n', noise), ('s', split)])
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
