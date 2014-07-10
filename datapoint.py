import numpy
import sys
import csv
from scipy.optimize import curve_fit
from itertools import groupby, chain
import random

import plant

# number of data points after every stimulus to use
window_size = 2600

# offset of window from start of stimulus (positive = after)
window_offset = -600

# min offset of null data from start of readings and first stimuli
null_offset = 600

def generate(plant_data):
    """
    Process plant data to produce a list of classified data points.
    """

    # if sample rate is not close to ideal sample rate, drop this data
    if plant.ideal_freq / plant_data.sample_freq < 0.9:
        print "Dropping data %s, bad sample rate" % plant_data.name
        return None

    X = []
    y = []

    def add_window(start, stim_type):
        window = plant_data.readings[start:start+window_size]

        # skip if window is not large enough (e.g. stimulus near end of data)
        if len(window) == window_size:
            X.append(window)
            y.append(stim_type)

    for stim in plant_data.stimuli:
        # create a window on each stimulus
        add_window(stim.time + window_offset, stim.type)

    first_stim = min([s.time for s in plant_data.stimuli])

    # get null stimulus from windows before the first stimulus
    null_start = null_offset
    while null_start + window_size + null_offset < first_stim:
        add_window(null_start, 'null')
        null_start += window_size

    return X, y, [plant_data.name] * len(X)


def generate_all(plants):
    """
    Process a list of plant data.

    Params:
        plants: A list of PlantData
    """
    X = []
    y = []
    sources = []

    for plant_data in plants:
        result = generate(plant_data)
        if result is None:
            continue
        Xp, yp, sourcep = result
        X += Xp
        y += yp
        sources += sourcep

    return X, y, sources


def save(path, X, y, sources):
    """
    Save data points to a CSV file.

    Params:
        path: File to save data points to.
    """

    with file(path, 'w') as f:
        writer = csv.writer(f)
        for xx, yy, source in zip(X, y, sources):
            # write a row for every data point
            writer.writerow([yy, source] + xx.T.flatten().tolist())


def load(path):
    """
    Load data points from a csv file.

    Params:
        path: File to load data points from.
    """

    X = []
    y = []

    with file(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            xx = map(float, row[1:])
            xx = numpy.array(xx)
            xx = numpy.reshape(xx, (2, -1)).T  # reshape into two columns
            X.append(xx)
            y.append(row[0])

    return X, y

def filter_types(X, y, types):
    """
    Params:
        classes: The allowed stimulus types.
    Returns: All datapoints of the given types.
    """
    return zip(*filter((lambda d: d[1] in types), zip(X, y)))

def group_types(X, y):
    """ Returns: The datapoints grouped together by type. """

    def by_type(d):
        return d[1]

    groups = groupby(sorted(zip(X, y), key=by_type), key=by_type)
    return [(yy, zip(*g)) for yy, g in groups]

def sample(X, y, group_size):
    # if class is too small, duplicate data and sample remainder
    # if class is too big, duplicate will be empty, take random sample
    duplicate = zip(X, y) * int(group_size / len(X))
    sample = random.sample(zip(X, y), group_size % len(X))
    return zip(*(duplicate + sample))

def balance(X, y, undersample=True):
    """
    Params:
        undersample:
            True to reduce the size of common classes, false to 
            replicate datapoints in uncommon classes.
    Returns: A subset with the same number of every represented type.
    """
    
    # find smallest datapoint type to decide how to balance
    all_sizes = [len(list(ys)) for yy, (Xs, ys) in group_types(X, y)]
    if undersample:
        group_size = min(all_sizes)
    else:
        group_size = max(all_sizes)

    # pick a random sample from each group
    samples = [sample(Xs, ys, group_size) for yy, (Xs, ys) in group_types(X, y)]
    X, y = zip(*samples)

    # concatenate all samples and return
    return (numpy.array(list(chain(*X))), numpy.array(list(chain(*y))))
