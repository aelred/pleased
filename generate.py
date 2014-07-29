import plant
import datapoint
import plot
import transform

from sklearn import pipeline
import numpy as np

# load all plant data in directory
print "Loading data"
plants = plant.load_all()

# process all data
print "Processing data"
X, y, sources = datapoint.generate_all(plants)

# write data to file
print "Writing to data.csv"
datapoint.save("data.csv", X, y, sources)

concat = transform.ConcatTransform()
split = transform.SplitTransform(divs=2)


def plot_plants():
    plot.plant_data_save(plants)


def plot_datapoints():
    plot.datapoints_save(X, y)


def plot_detrended():
    # plot detrended points
    detrend = transform.MapTransform(transform.DetrendTransform(), divs=2)
    pipe = pipeline.Pipeline([('c', concat), ('d', detrend), ('s', split)])
    plot.datapoints_save(pipe.transform(X), y, 'detrend')


def plot_wavelets():
    # plot wavelets
    wavelets = transform.DiscreteWaveletTransform('haar', 11, 0)
    avg = transform.ElectrodeAvgTransform()
    plot.datapoints_save(wavelets.transform(avg.transform(X)), y,
                         'wavelet', plot.datapoint_set)


mov_avg = transform.MapTransform(transform.MovingAvgTransform(100), divs=2)
deriv = transform.MapTransform(transform.differential, divs=2)
mean = transform.MapTransform(transform.MeanSubtractTransform(), divs=2)


def plot_derivatives():
    # plot first derivatives
    pipe = pipeline.Pipeline(
        [('c', concat), ('m', mov_avg), ('d', deriv), ('me', mean), ('s', split)])
    plot.datapoints_save(pipe.transform(X), y, 'deriv')


def plot_derivatives_abs():
    # plot first derivatives absolute value
    abs_ = transform.Extractor(np.absolute)
    pipe = pipeline.Pipeline(
        [('c', concat), ('m', mov_avg), ('d', deriv),
         ('me', mean), ('a', abs_), ('s', split)])
    plot.datapoints_save(pipe.transform(X), y, 'deriv_abs')
