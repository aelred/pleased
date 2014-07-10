from sklearn import base, svm, lda, qda, pipeline, preprocessing, grid_search
from transform import *
import pywt
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import chain, groupby

import plant
import datapoint


labels = ['null', 'ozone', 'H2SO4']


def preprocess(plants):
    # extract windows from plant data
    X, y, sources = datapoint.generate_all(plants)
    # filter to relevant datapoint types
    X, y = datapoint.filter_types(X, y, labels)
    # balance the dataset
    X, y = datapoint.balance(X, y, False)
    
    # take the average and detrend the data ahead of time
    X = ElectrodeAvgTransform().transform(X)
    X = DetrendTransform().transform(X)
    X = PostStimulusTransform(60).transform(X)

    return X, y


def plot_features(f1, f2):
    # load plant data from files
    plants = plant.load_all()
    # preprocess data
    X, y = preprocess(plants)

    # scale data
    X = FeatureEnsembleTransform().transform(X)
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    groups = lambda: datapoint.group_types(X, y)

    # visualize the feature extractor
    colors = iter(cm.rainbow(np.linspace(0, 1, len(list(groups())))))
    for dtype, (Xg, yg) in groups():
        plt.scatter(Xg[:,f1], Xg[:,f2], c=next(colors), label=dtype)
    plt.legend()
    plt.show()


def plot_histogram(feature):
    # load plant data from files
    plants = plant.load_all()
    # preprocess data
    X, y = preprocess(plants)

    groups = lambda: datapoint.group_types(X, y)

    # visualize a histogram of the feature
    for dtype, (Xg, yg) in groups():
        Xg = FeatureEnsembleTransform().transform(Xg)
        plt.hist(Xg[:,feature], bins=40, alpha=0.5, label=dtype)
    plt.legend()
    plt.show()


_ensemble = FeatureEnsembleTransform().extractor
_window = WindowTransform(_ensemble, 3, False).extractor
pre_pipe = [
    ('feature', DecimateWindowTransform(_window)),
    ('scaler', preprocessing.StandardScaler())
]


def plot_pipeline():
    # load plant data from files
    plants = plant.load_all()
    # preprocess data
    X, y = preprocess(plants)

    # transform data on pipeline
    lda_ = lda.LDA(2)
    lda_pipe = pipeline.Pipeline(pre_pipe + [('lda', lda_)])
    lda_pipe.fit(X, y)
    yp = lda_pipe.predict(X)
    X = lda_pipe.transform(X)

    groups = datapoint.group_types(zip(X, yp), y)

    # visualize the pipeline 
    colors = iter(cm.rainbow(np.linspace(0, 1, len(list(groups)))))
    for dtype, (Xg, yg) in groups:
        # extract predicted class
        Xg, yp = map(np.array, zip(*Xg))
        tp = (yg == yp)
        Xtp, Xfp = Xg[tp], Xg[~tp]  # find true and false positives
        c = next(colors)
        plt.scatter(Xtp[:,0], Xtp[:,1], marker='o', c=c, label=dtype)
        plt.scatter(Xfp[:,0], Xfp[:,1], marker='x', c=c, 
                    label=dtype + " false positive")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # load plant data from files
    plants = plant.load_all()

    # split plant data into training and validation sets
    random.shuffle(plants)
    train_len = int(0.75 * len(plants))
    train_plants = plants[:train_len]
    valid_plants = plants[train_len:]

    print "Experiments in training set:", len(train_plants)
    print "Experiments in validation set:", len(valid_plants)

    # get X data and y labels
    X_train, y_train = preprocess(train_plants)
    X_valid, y_valid = preprocess(valid_plants)

    print "Datapoints in training set:", len(X_train)
    class_train = [(d[0], len(list(d[1]))) for d in groupby(y_train)]
    print "Classes in training set:", class_train 
    print "Datapoints in validation set:", len(X_valid)
    class_valid = [(d[0], len(list(d[1]))) for d in groupby(y_valid)]
    print "Classes in validation set:", class_valid

    # set up pipeline
    pipeline = pipeline.Pipeline(pre_pipe + [('svm', svm.SVC())])
    params = [{}]

    # perform grid search on pipeline, get best parameters from training data
    grid = grid_search.GridSearchCV(pipeline, params, cv=5, verbose=2)
    grid.fit(X_train, y_train)
    classifier = grid.best_estimator_

    print "Grid search results:"
    print grid.best_score_

    # test the classifier on the validation data set
    validation_score = classifier.fit(X_train, y_train).score(X_valid, y_valid)

    print "Validation data results:"
    print validation_score
