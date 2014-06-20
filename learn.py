from sklearn.cross_validation import train_test_split

import plant
import datapoint

if __name__ == "__main__":
	# load plant data from files
	plants = plant.load_all()

	# split plant data into training and validation sets
	train_plants, valid_plants = train_test_split(plants)

	def preprocess(plants):
		# extract windows from plant data
		datapoints = generate_all(plants)
		# balance the dataset
		balanced = datapoint.balance(datapoints)

		# extract features and labels
		labels = [d[0] for d in datapoints]
		data = [d[1] for d in datapoints]
		return data, labels

	train = preprocess(train_plants)
	valid = preprocess(valid_plants)

	# TODO: Split train into sub-train/test sets

	# TODO: Learn parameters on sub-train/test sets

	# TODO: Validate on train/validate sets