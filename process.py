import numpy
import sys
import csv

import plant
import plot

# number of data points after every stimulus to use
window_size = 5000

# offset of window from start of stimulus (positive = after)
window_offset = 0


def process(plant_data):
    """
    Process plant data to produce a list of classified data points.

    Returns:
        A list of tuples, where the first element of the tuple is the
        type of stimulus and the second element is the data.
    """
    new_data = []

    for stim in plant_data.stimuli:
        # create a window on each stimulus
        start = stim.time + window_offset
        window = plant_data.readings[start:start+window_size]

        # center around starting value of window
        window = numpy.array([w - window[0] for w in window])

        # skip if window is not large enough (e.g. stimulus near end of data)
        if len(window) != window_size:
            continue

        new_data.append((stim.type, window))

    return new_data


def process_all(plants):
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
        new_data += process(plant_data)

    return new_data


def save_datapoints(path, datapoints):
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


def load_datapoints(path):
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


# if called from command line, load mat files from given path
if __name__ == "__main__":
    # get directory from command line
    path = sys.argv[1]

    # load all plant data in directory
    print "Loading data"
    plants = plant.load_all(path)

    # process all data
    print "Processing data"
    datapoints = process_all(plants)

    # write data to file
    print "Writing to data.csv"
    save_datapoints("data.csv", datapoints)

    # create plots of each datapoints
    print "Creating plots"
    plot.save_datapoint_plots(datapoints)