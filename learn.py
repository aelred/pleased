from sklearn.cross_validation import train_test_split

import plant
import datapoint

if __name__ == "__main__":
	# load plant data from files
	plants = plant.load_all()

	# split plant data into training and validation sets
	train_plants, valid_plants = train_test_split(plants)

	# extract windows from plant data
	train = generate_all(train_plants)
	valid = generate_all(valid_plants)

	# TODO: balance datasets

	# TODO: Split train into sub-train/test sets

	# TODO: Learn parameters on sub-train/test sets

	# TODO: Validate on train/validate sets