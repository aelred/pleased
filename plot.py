import matplotlib.pyplot as plt
from collections import Counter


def plot_plant_data(plant_data):
	plt.plot(plant_data.readings)
	for s in plant_data.stimuli:
		plt.axvline(s.time)

def plot_datapoints(datapoints):
	map(plot_datapoint, datapoints)

def plot_datapoint(datapoint):
	plt.plot(datapoint[1])

def show():
	plt.show()

def save_datapoint_plots(datapoints):
	type_count = Counter()

	for d in datapoints:
		type_count[d[0]] += 1
		plot_datapoint(d)
		plt.savefig("plots/%s%d.png" % (d[0], type_count[d[0]]))
		plt.clf()