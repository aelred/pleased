import matplotlib.pyplot as plt


def plot_plant_data(plant_data):
	plt.plot(plant_data.readings)
	plt.axvline([s.time for s in plant_data.stimuli])
	plt.show()

def plot_datapoints(datapoints):
	for d in datapoints:
		plt.plot(d[1])
	plt.show()

def plot_datapoint(datapoint):
	plt.plot(datapoint[1])
	plt.show()