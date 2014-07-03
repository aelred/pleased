import numpy
import sys
import csv
from scipy.optimize import curve_fit
from scipy.signal import decimate
from itertools import groupby, chain
import random

# number of data points after every stimulus to use
window_size = 2600

# offset of window from start of stimulus (positive = after)
window_offset = -600

# min offset of null data from start of readings and first stimuli
null_offset = 600

# ideal sample frequency for generated datapoints
sample_freq = 1.0


def generate(plant_data):
    """
    Process plant data to produce a list of classified data points.

    Returns:
        A list of tuples, where the first element of the tuple is the
        type of stimulus and the second element is the data.
    """

    dec_factor = int(sample_freq / plant_data.sample_freq)

    # if sample rate is not close to ideal sample rate, drop this data
    if dec_factor < 0.9:
        print "Dropping data %s, bad sample rate" % plant_data.name
        return []

    new_data = []

    def add_window(start, stim_type):
        window = plant_data.readings[start:start+window_size*dec_factor]
        if dec_factor > 1.0:
            # resample the window to the right frequency
            window = decimate(window, dec_factor, ftype='fir', axis=0)
        datapoint = (stim_type, window)

        # skip if window is not large enough (e.g. stimulus near end of data)
        if len(window) == window_size:
            new_data.append(datapoint)

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

    Params:
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

    Params:
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

    Params:
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

def filter_types(datapoints, types):
    """
    Params:
        datapoints: A list of datapoints to filter.
        classes: The allowed stimulus types.
    Returns: All datapoints of the given types.
    """
    return filter((lambda d: d[0] in types), datapoints)

def group_types(datapoints):
    """
    Params:
        datapoints: A list of datapoints to group.
    Returns: The datapoints grouped together by type.
    """

    def by_type(d):
        return d[0]

    return groupby(sorted(datapoints, key=by_type), key=by_type)

def balance(datapoints):
    """
    Params:
        datapoints: A list of datapoints to balance.
    Returns: A subset with the same number of every represented type.
    """

    # find smallest datapoint type to decide how to balance
    group_size = min(len(list(g[1])) for g in group_types(datapoints))

    # pick a random sample from each group
    samples = [random.sample(list(g[1]), group_size) 
               for g in group_types(datapoints)]

    # concatenate all samples and return
    return list(chain(*samples))
