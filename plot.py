import matplotlib.pyplot as plt
from collections import Counter
import os
import os.path
import glob

import datapoint as datap


def plant_data(pd):
    plt.plot(pd.readings)
    for s in pd.stimuli:
        plt.axvline(s.time)


def datapoints(X, y):
    [datapoint(xx, yy) for xx, yy in zip(X, y)]


def datapoint(xx, yy, stim_line=True):
    plt.plot(xx)
    if stim_line:
        plt.axvline(-datap.window_offset)


def datapoint_set(XX, yy):
    fig, axes = plt.subplots(nrows=len(XX), ncols=1)
    for xx, ax in zip(XX, axes):
        ax.plot(xx)


def show():
    plt.show()


def plant_data_save(plant_list, path):
    # empty folder in advance
    files = glob.glob(path)
    for f in files:
        if os.path.isfile(f):
            os.remove(f)

    for p in plant_list:
        plant_data(p)
        plt.savefig(os.path.join("plots", path, "%s.jpg" % p.name))
        plt.clf()
        plt.close()


def datapoints_save(X, y, path, plot_func=datapoint):
    type_count = Counter()

    # empty folder in advance
    files = glob.glob(path)
    for f in files:
        if os.path.isfile(f):
            os.remove(f)

    for xx, yy in zip(X, y):
        type_count[yy] += 1
        plot_func(xx, yy)
        plt.savefig(os.path.join("plots", path, "%s_%d.png" % (yy, type_count[yy])))
        plt.clf()
        plt.close()
