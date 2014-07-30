import plant
import datapoint
import plot
import transform

from sklearn import pipeline
import numpy as np
import scipy

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
deriv = transform.Map(transform.differential, divs=2)
mean = transform.Map(transform.MeanSubtract(), divs=2)


def plot_derivatives():
    # plot first derivatives
    pipe = pipeline.Pipeline(
        [('c', concat), ('m', mov_avg), ('d', deriv), ('me', mean), ('s', split)])
    plot.datapoints_save(pipe.transform(X), y, 'deriv')

abs_ = transform.Extractor(np.absolute)


def plot_derivatives_abs():
    # plot first derivatives absolute value
    pipe = pipeline.Pipeline(
        [('c', concat), ('m', mov_avg), ('d', deriv),
         ('me', mean), ('a', abs_), ('s', split)])
    plot.datapoints_save(pipe.transform(X), y, 'deriv_abs')

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
