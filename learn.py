from sklearn import base, svm, lda, qda, pipeline, preprocessing, grid_search
from transform import *
import pywt
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.lines as lines
from itertools import chain, groupby

import plant
import datapoint


def_labels = ['null', 'ozone', 'H2SO4']


class Classifier:

    def __init__(self, preproc_pipe, extract_pipe, 
                 postproc_pipe, classifier, labels=None, params=None):
        self.preproc_pipe = preproc_pipe
        self.extract_pipe = extract_pipe
        self.postproc_pipe = postproc_pipe
        self.classifier = classifier
        self.params = params or [{}]
        self.labels = labels or def_labels

    def get_data(self, plants=None):
        # load plants if parameter not provided
        if plants is None:
            plants = plant.load_all()
        # extract windows from plant data
        X, y, sources = datapoint.generate_all(plants)
        # filter to relevant datapoint types
        X, y = datapoint.filter_types(zip(X, sources), y, self.labels)
        # balance the dataset
        X, y = datapoint.balance(X, y, False)

        X, sources = zip(*X)

        return self.preprocess(np.array(X), np.array(y), np.array(sources))

    def preprocess(self, X, y, sources):
        return pipeline.Pipeline(self.preproc_pipe).fit_transform(X, y), y, sources

    def _lda(self, dim=None):
        # load and preprocess data
        X, y, sources = self.get_data()

        # transform data on pipeline
        lda_ = lda.LDA(dim)
        lda_pipe = pipeline.Pipeline(
            self.extract_pipe + self.postproc_pipe + [('lda', lda_)])
        lda_pipe.fit(X, y)
        print lda_.scalings_.shape
        yp = lda_pipe.predict(X)
        X = lda_pipe.transform(X)

        return X, y, yp, lda_

    def plot_lda_scaling(self):
        X, y, yp, lda_ = self._lda()
        plt.plot(lda_.scalings_)
        plt.show()

    def _plot(self, dim, title, fig_func, plt_func):
        # transform data by linear discriminant analysis
        X, y, yp = self._lda(dim)

        groups = datapoint.group_types(zip(X, yp), y)

        # visualize the pipeline 
        colors = iter(cm.rainbow(np.linspace(0, 1, len(list(groups)))))
        fig, axes = fig_func()
        for dtype, (Xg, yg) in groups:
            # extract predicted class
            Xg, yp = map(np.array, zip(*Xg))
            tp = (yg == yp)
            Xtp, Xfp = Xg[tp], Xg[~tp]  # find true and false positives
            c = next(colors)

            plt_func(axes, Xtp, marker='o', c=c, label=dtype)
            plt_func(axes, Xfp, marker='x', c=c, label=dtype + ' fp')

        axes.set_xlabel('LDA Basis vector 1')
        axes.set_ylabel('LDA Basis vector 2')
        if title:
            axes.set_title(title)
        axes.legend()
        fig.show()

    def plot(self, title=None):
        def plt_func(axes, X, marker, c, label):
            axes.scatter(X[:, 0], X[:, 1], marker=marker, c=c, label=label)
        self._plot(2, title, plt.subplots, plt_func)

    def plot3d(self, title=None):
        def fig_func():
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
            axes.set_zlabel('LDA Basis vector 3')
            return fig, axes
        def plt_func(axes, X, marker, c, label):
            axes.scatter(X[:, 0], X[:, 1], X[:, 2], 
                         marker=marker, c=c, label=label)
            # proxy plot to appear on legend
            axes.plot([0],[0],linestyle='none', 
                         marker=marker, c=c, label=label)

        self._plot(3, title, fig_func, plt_func)

    def score(self):
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
        X_train, y_train = self.get_data(train_plants)
        X_valid, y_valid = self.get_data(valid_plants)

        print "Datapoints in training set:", len(X_train)
        class_train = [(d[0], len(list(d[1]))) for d in groupby(y_train)]
        print "Classes in training set:", class_train 
        print "Datapoints in validation set:", len(X_valid)
        class_valid = [(d[0], len(list(d[1]))) for d in groupby(y_valid)]
        print "Classes in validation set:", class_valid

        # set up pipeline
        pipe = pipeline.Pipeline(
            self.extract_pipe + self.postproc_pipe + 
            [('classifier', self.classifier)])

        # perform grid search on pipeline, get best parameters from training data
        grid = grid_search.GridSearchCV(pipe, self.params, cv=5, verbose=2)
        grid.fit(X_train, y_train)
        classifier = grid.best_estimator_

        print "Grid search results:"
        print grid.best_score_

        # test the classifier on the validation data set
        validation_score = classifier.fit(X_train, y_train).score(X_valid, y_valid)

        print "Validation data results:"
        print validation_score


class NullClassifier(Classifier):

    def preprocess(self, X, y, sources):
        """ 
        Method that changes null class labels to instead be the source of the null.
        This allows testing if null data can be distinguished per-experiment.
        """
        def set_class(yy, source):
            if yy == 'null':
                return 'null ' + source.split("#")[0]
            else:
                return yy
        y = [set_class(yy, source) for yy, source in zip(y, sources)]
        return Classifier.preprocess(self, X, y, sources)
