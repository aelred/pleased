from collections import namedtuple
import scipy.io
import numpy
import os
import csv
import cPickle
from scipy.signal import decimate

stim_types = {
    'water': ['acqua piante'],
    'H2SO4': ['h2so4'],
    'ozone': ['ozone', 'ozono', 'o3'],
    'NaCL': ['nacl'],
    'light-on': ['light-on'],
    'light-off': ['light-off']
}

# the ideal sample frequency to sample the plant data at
ideal_freq = 0.1

# a stimulus on the plant, defined as a type (such as 'ozone') and time
# the null (no) stimulus is named 'null'
Stimulus = namedtuple('Stimulus', ['type', 'time'])

# data on a single experiment on a single plant
# readings is a 2D array where each column relates to an electrode on the plant
PlantData = namedtuple('PlantData',
                       ['name', 'readings', 'stimuli', 'sample_freq'])

# pre-loaded plant data
_plant_data = None


def load_all(path="."):
    """
    Load all plant data from .txt files.

    Args:
        path: Path to a directory.
    Returns: A list of PlantData
    """

    # first check if data has already been loaded (for super-quick loading!)
    global _plant_data
    if _plant_data:
        return _plant_data

    # first check for previously generated plant data (for quicker loading)
    plant_file = os.path.join(path, "plant_data")
    if os.path.isfile(plant_file):
        with file(plant_file, 'r') as f:
            print "Loading from generated file %s" % plant_file
            _plant_data = cPickle.load(f)
            return _plant_data

    plants = []

    for root, dirs, files in os.walk(path):
        if "blk0" in dirs:
            print "Reading %s" % root
            plants += load_txt(root)

    # save generated plant data to pickle file for faster loading
    print "Writing to pickle file"
    with file(plant_file, 'w') as f:
        cPickle.dump(plants, f)

    _plant_data = plants
    return _plant_data


def load_txt(path):
    """
    Load plant data from .txt files.

    Args:
        path: Path to a data folder.
    Returns: A list of PlantData
    """
    raw_data = []
    stimuli = []

    i = 0
    mark_offset = 0

    # read sample frequency from settings file
    with file(os.path.join(path, "blk0", "blk_setting.txt"), 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row[0] == 'SPEED ':
                sample_freq = 1.0 / float(row[1])
                break

    while os.path.exists(os.path.join(path, "blk%d" % i)):

        blk = os.path.join(path, "blk%d" % i)

        # check common data file locations
        data = os.path.join(blk, "data2.txt")
        if not os.path.isfile(data):
            data = os.path.join(blk, "data.txt")

        marks = os.path.join(blk, "marks.txt")

        with file(marks, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # skip header
            for row in reader:
                stimuli.append(Stimulus(row[0].strip(),
                                        int(row[2].strip()) + mark_offset))

        with file(data, 'r') as f:
            reader = csv.reader(f, delimiter='\t')

            for row in reader:
                new_data = row[:-1]  # skip last column

                # skip empty rows
                if len(new_data) == 0:
                    continue

                try:
                    raw_data.append(map(float, new_data))
                    mark_offset += 1
                except ValueError:
                    # catch case where looking at header
                    pass

        i += 1

    fname = os.path.basename(path)
    # WHY IS ALL THE DATA STRUCTURED DIFFERENTLY???
    if fname == 'Electrical signal':
        fname = os.path.basename(os.path.split(path)[0])

    return format_raw(fname, raw_data, stimuli, sample_freq)


def load_mat(path):
    """
    Load plant data from a .mat file.

    Args:
        path: Path to a .mat file.
    Returns: A list of PlantData
    """
    mat = scipy.io.loadmat(path)

    # get astonishingly poorly-named matrix of readings
    readings = mat['b\x001\x00\x00\x00']

    # calculate sample frequency
    total_time = readings[-1][0] - readings[0][0]
    sample_freq = total_time / len(readings)
    # TODO: Worry about when the sample frequency is different (interpolate?)

    # get all labelled stimuli
    stimuli = []
    i = 0
    while 'm%03d' % i in mat:
        # get name and time value of stimulus in terrifyingly deep array WTF
        stim_data = mat['m%03d' % i]
        name = stim_data[0][0][1][0]
        time = stim_data[0][0][0][0][0]

        # calculate index of readings array from time and time step per reading
        index = time / sample_freq

        stimuli.append(Stimulus(name, index))

        i += 1

    fname = os.path.basename(path)

    return format_raw(fname, readings[:, 1:], stimuli, sample_freq)


def format_raw(name, raw_data, raw_stimuli, sample_freq):
    """
    Process raw data from a file into plant data.

    Args:
        name: The name of the plant data.
        raw_data: A 2D array of readings with no time information.
        raw_stimuli: A list of Stimulus which are not necessarily valid.
    Returns: A list of PlantData
    """
    stimuli = []

    readings = numpy.array(raw_data)
    print readings.shape

    for stim in raw_stimuli:
        # find type of stimulus
        type_ = None

        if 'stop' in stim.type.lower():
            # this is the end of a stimulus, don't use
            continue

        for t, aliases in stim_types.iteritems():
            for alias in aliases:
                if alias in stim.type.lower():
                    type_ = t
                    break

            if type_:
                break

        # if type recognized, add to stimuli
        if type_ is not None:
            stimuli.append(Stimulus(type_, stim.time))

    # for every pair of readings, create a plant data object
    plants = []
    for i, (r1, r2) in enumerate(zip(readings.T[0::2], readings.T[1::2])):
        data = numpy.array([r1, r2]).T
        plant = PlantData("%s_%d" % (name, i), data, stimuli, sample_freq)
        plant = resample(plant, ideal_freq)
        plants.append(plant)

    return plants


def resample(plant_data, new_sample_freq):
    """
    Resample some plant data to a new, lower sampling frequency.

    Args:
        plant_data: The plant data to resample.
        sample_freq: The new frequency, must not be higher than the current frequency.
    Returns: A new plant data at the new frequency.
    """
    if new_sample_freq <= plant_data.sample_freq:
        return plant_data

    dec_factor = int(new_sample_freq / plant_data.sample_freq)
    readings = decimate(plant_data.readings, dec_factor, ftype='fir', axis=0)
    stimuli = [Stimulus(s.type, s.time / dec_factor) for s in plant_data.stimuli]
    return PlantData(plant_data.name, readings, stimuli, new_sample_freq)
