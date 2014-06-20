import numpy
import sys
import csv
from scipy.optimize import curve_fit
from itertools import groupby, chain
import random

# number of data points after every stimulus to use
window_size = 12000

# offset of window from start of stimulus (positive = after)
window_offset = -6000

# min offset of null data from start of readings and first stimuli
null_offset = 6000

# ideal sample frequency for generated datapoints
sample_freq = 0.1


def generate(plant_data):
    """
    Process plant data to produce a list of classified data points.

    Returns:
        A list of tuples, where the first element of the tuple is the
        type of stimulus and the second element is the data.
    """

    # if sample rate is not close to ideal sample rate, drop this data
    if abs(plant_data.sample_freq - sample_freq) > 0.1:
        print "Dropping data %s, bad sample rate" % plant_data.name
        return []

    new_data = []

    def add_window(start, stim_type):
        window = plant_data.readings[start:start+window_size]

        # center around starting value of window
        window = numpy.array([w - window[0] for w in window])

        # skip if window is not large enough (e.g. stimulus near end of data)
        if len(window) == window_size:
            new_data.append((stim_type, window))

    for stim in plant_data.stimuli:
        # create a window on each stimulus
        add_window(stim.time + window_offset, stim.type)

    first_stim = min([s.time for s in plant_data.stimuli])

    # get null stimulus from windows before the first stimulus
    null_start = null_offset
    while null_start + window_size + null_offset < first_stim:
        add_window(null_start, 'null')
        null_start += window_size

    return new_data


def generate_all(plants):
    """
    Process a list of plant data.

    Args:
        plants: A list of PlantData
    Returns:
        A list of tuples, where the first element of the tuple is the type of
        stimulus and the second element is the data.
    """
    new_data = []

    for plant_data in plants:
        new_data += generate(plant_data)

    return new_data


def save(path, datapoints):
    """
    Save data points to a CSV file.

    Args:
        path: File to save data points to.
        datapoints:
            A list of tuples, where the first element of the tuple is the type
            of stimulus and the second element is the data.
    """

    with file(path, 'w') as f:
        writer = csv.writer(f)
        for stim_type, data in datapoints:
            # write a row for every data point
            writer.writerow([stim_type] + data.T.flatten().tolist())


def load(path):
    """
    Load data points from a csv file.

    Args:
        path: File to load data points from.
    Returns:
        A list of tuples, where the first element of the tuple is the type of
        stimulus and the second element is the data.
    """

    datapoints = []

    with file(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            stim_type = row[0]
            data = map(float, row[1:])
            data = numpy.array(data)
            data = numpy.reshape(data, (2, -1)).T  # reshape into two columns
            datapoints.append((stim_type, data))

    return datapoints


def detrend(datapoint):
    """
    Eliminate linear trends in a window by examining time before stimulus occurs.

    Args:
        datapoint: The datapoint to detrend.
    Returns: A new detrended datapoint.
    """
    data = datapoint[1]

    def linear(xs, m, c):
        return map(lambda x: m*x + c, xs)

    def detrend_column(col):
        # find best fitting curve to pre-stimulus window
        times = range(0, len(col))
        params, cov = curve_fit(linear, times[0:-window_offset], 
                                col[0:-window_offset], (0, 0))
        # subtract extrapolated curve from data to produce new dataset
        return col - linear(times, *params)

    de_data = numpy.array([detrend_column(data[:,0]), detrend_column(data[:,1])])

    return (datapoint[0], de_data.T)


def balance(datapoints):
    """
    Args:
        datapoints: A list of datapoints to balance.
    Returns: A subset with the same number of every represented type.
    """

    def by_type(d):
        return d[0]

    # group datapoints by type
    def groups():
        return groupby(sorted(datapoints, key=by_type), key=by_type)

    # find smallest datapoint type to decide how to balance
    group_size = min(len(list(g[1])) for g in groups())
    print group_size

    # pick a random sample from each group
    samples = [random.sample(list(g[1]), group_size) for g in groups()]

    # concatenate all samples and return
    return list(chain(*samples))