import matplotlib.pyplot as plt


def plot_plant_data(plant_data):
	plt.plot(plant_data.readings)
	plt.axvline([s.time for s in plant_data.stimuli])

def plot_datapoints(datapoints):
	map(plot_datapoint, datapoints)

def plot_datapoint(datapoint):
	plt.plot(datapoint[1])
	plt.show()