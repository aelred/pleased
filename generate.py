import plant
import datapoint
import plot

# if called from command line, load mat files from data directory
if __name__ == "__main__":
    # load all plant data in directory
    print "Loading data"
    plants = plant.load_all()

    # process all data
    print "Processing data"
    datapoints = datapoint.generate_all(plants)

    # write data to file
    print "Writing to data.csv"
    datapoint.save("data.csv", datapoints)

    # create plots of each datapoint and plant
    print "Creating plots"
    plot.plant_data_save(plants)
    plot.datapoints_save(datapoints)