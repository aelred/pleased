import matplotlib.pyplot as plt
from collections import Counter


def plant_data(pd):
	plt.plot(pd.readings)
	for s in pd.stimuli:
		plt.axvline(s.time)

def datapoints(ds):
	map(datapoint, ds)

def datapoint(d):
	plt.plot(d[1])

def show():
	plt.show()

def plant_data_save(plant_list):
	for p in plant_list:
		plant_data(p)
		plt.savefig("plant_plots/%s.jpg" % p.name)
		plt.clf()

def datapoints_save(datapoints):
	type_count = Counter()

	for d in datapoints:
		type_count[d[0]] += 1
		datapoint(d)
		plt.savefig("plots/%s%d.png" % (d[0], type_count[d[0]]))
		plt.clf()