import matplotlib.pyplot as plt
from collections import Counter
import os

import datapoint as datap


def plant_data(pd):
	plt.plot(pd.readings)
	for s in pd.stimuli:
		plt.axvline(s.time)

def datapoints(X, y):
    [datapoint(xx, yy) for xx, yy in zip(X, y)]

def datapoint(xx, yy):
    plt.plot(xx)
    plt.axvline(-datap.window_offset)

def show():
	plt.show()

def plant_data_save(plant_list, path="plant_plots"):
	for p in plant_list:
		plant_data(p)
		plt.savefig(os.path.join(path, "%s.jpg" % p.name))
		plt.clf()

def datapoints_save(X, y, path="plots"):
	type_count = Counter()

	for xx, yy in zip(X, y):
		type_count[yy] += 1
		datapoint(xx, yy)
		plt.savefig(os.path.join(path, "%s_%d.png" % (yy, type_count[yy])))
		plt.clf()
