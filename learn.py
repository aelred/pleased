from sklearn import lda, pipeline, grid_search
from transform import *
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from itertools import groupby

import plant
import datapoint


def_labels = ['null', 'ozone', 'H2SO4']


def _scatter(plt_func, axes, X, y, yp, label, mark_tp, mark_fp):
    groups = datapoint.group_types(zip(X, yp), y)

    # select a rainbow of colours
    colors = iter(cm.rainbow(np.linspace(0, 1, len(list(groups)))))
    scatters = []
    for dtype, (Xg, yg) in groups:
        # extract predicted class
        Xg, yp = map(np.array, zip(*Xg))
        yp = np.array(map(str, yp))
        yg = np.array(map(str, yg))
        tp = yg == yp
        Xtp, Xfp = Xg[tp], Xg[~tp]  # find true and false positives
        c = next(colors)

        t_label = label + str(dtype)
        plot_tp = plt_func(axes, Xtp, marker=mark_tp, c=c, label=t_label)
        plot_fp = plt_func(axes, Xfp, marker=mark_fp, c=c, label=None)
        scatters.append((plot_tp, plot_fp))

    return scatters


class Classifier:

    def __init__(self, preproc_pipe, extract_pipe,
                 postproc_pipe, classifier,
                 lda_=None, labels=None, params=None):
        self.preproc_pipe = preproc_pipe
        self.extract_pipe = extract_pipe
        self.postproc_pipe = postproc_pipe
        self.classifier = classifier
        self.lda = lda_ or lda.LDA()
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
        return (pipeline.Pipeline(self.preproc_pipe).fit_transform(X, y),
                y, sources)

    def _split_data(self, plants=None):
        # load plants if parameter is not provided
        if plants is None:
            plants = plant.load_all()

        # split plant data into training and validation sets
        random.shuffle(plants)
        train_len = int(0.75 * len(plants))
        train_plants = plants[:train_len]
        valid_plants = plants[train_len:]

        # get X data and y labels
        X_train, y_train, source_train = self.get_data(train_plants)
        X_valid, y_valid, source_valid = self.get_data(valid_plants)
        return X_train, X_valid, y_train, y_valid, source_train, source_valid

    def _run_lda(self, split=True):
        # load and preprocess data
        if split:
            X_train, X_valid, y_train, y_valid, _, _ = self._split_data()
        else:
            X_train, y_train, _ = self.get_data()
            X_valid, y_valid = X_train, y_train

        # transform data on pipeline
        lda_pipe = pipeline.Pipeline(
            self.extract_pipe + self.postproc_pipe + [('lda', self.lda)])
        lda_pipe.fit(X_train, y_train)

        yp_train = lda_pipe.predict(X_train)
        X_train = lda_pipe.transform(X_train)

        if split:
            yp_valid = lda_pipe.predict(X_valid)
            X_valid = lda_pipe.transform(X_valid)
            return X_train, X_valid, y_train, y_valid, yp_train, yp_valid
        else:
            return X_train, y_train, yp_train

    def plot_lda_scaling(self, barchart, title=None, labels=None):
        self._run_lda(split=False)
        data = np.sum(np.absolute(self.lda.scalings_), 1)
        if barchart:
            plt.bar(range(len(data)), data)
            plt.xticks([x+0.5 for x in range(len(data))], labels)
        else:
            plt.plot(data)
        if title:
            plt.title(title)
        plt.xlabel('Feature')
        plt.ylabel('Significance by LDA')
        plt.show()

    def _plot(self, title, fig_func, plt_func, split):
        # transform data by linear discriminant analysis
        if split:
            Xt, Xv, yt, yv, ypt, ypv = self._run_lda(True)
            train_label = 'train '
        else:
            Xt, yt, ypt = self._run_lda(False)
            train_label = ''

        fig, axes = fig_func()

        if split:
            scatters = _scatter(plt_func, axes, Xt, yt, ypt,
                                train_label, '+', 'x')
            scatters += _scatter(plt_func, axes, Xv, yv, ypv,
                                 'valid ', 'o', 's')
        else:
            scatters = _scatter(plt_func, axes, Xt, yt, ypt,
                                train_label, 'o', 's')

        axes.set_xlabel('LDA Basis vector 1')
        axes.set_ylabel('LDA Basis vector 2')
        if title:
            axes.set_title(title)

        legend = axes.legend(fancybox=True, bbox_to_anchor=(1.12, 1.0))
        scatter_legend = {}
        for leg_line, scatter in zip(legend.get_texts(), scatters):
            leg_line.set_picker(True)
            scatter_legend[leg_line] = scatter

        def on_pick(event):
            # hacky solution to weird matplotlib bug
            on_pick.flag = not on_pick.flag
            if on_pick.flag:
                return

            # allow hiding/showing scatter plots
            leg_line = event.artist
            scatter = scatter_legend[leg_line]

            # toggle visibility
            for s in scatter:
                vis = not s.get_visible()
                s.set_visible(vis)

            # set visibility on legend
            if vis:
                leg_line.set_alpha(1.0)
            else:
                leg_line.set_alpha(0.2)
            fig.canvas.draw()
        on_pick.flag = True

        fig.canvas.mpl_connect('pick_event', on_pick)

        plt.show()

    def plot(self, title=None, split=True):
        def plt_func(axes, X, marker, c, label):
            return axes.scatter(X[:, 0], X[:, 1], marker=marker, c=c, label=label)
        self._plot(title, plt.subplots, plt_func, split)

    def plot3d(self, title=None, split=True):
        def fig_func():
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
            axes.set_zlabel('LDA Basis vector 3')
            return fig, axes

        def plt_func(axes, X, marker, c, label):
            # proxy plot to appear on legend
            axes.plot([0], [0], linestyle='none',
                      marker=marker, c=c, label=label)
            return axes.scatter(X[:, 0], X[:, 1], X[:, 2],
                                marker=marker, c=c, label=label)

        self._plot(title, fig_func, plt_func, split)

    def plot1d(self, title=None, split=True):
        def plt_func(axes, X, marker, c, label):
            return axes.scatter(X[:, 0], X[:, 0], marker=marker, c=c, label=label)
        self._plot(title, plt.subplots, plt_func, split)

    def score(self):
        # split plant data into training and validation sets
        X_train, X_valid, y_train, y_valid, st, sv = self._split_data()

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
