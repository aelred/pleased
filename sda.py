""" Provides an implementation of Sparse Discriminant Analysis. """

from sklearn import lda, preprocessing
from sklearn.utils import check_arrays, column_or_1d
import numpy as np
import random
from mlabwrap import mlab


class SDA(lda.LDA):
    """
    Sparse Discriminant Analysis (SDA)

    A classifier that behaves like LDA while maintaining a 
    sparse coefficient matrix.
    In other words, the number of features used is minimized.
    """

    def fit(self, X, y, store_covariance=False, tol=1.0e-6):
        X, y = check_arrays(X, y, sparse_format='dense')
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors

        # Group means n_classes*n_features matrix
        means = []
        Xc = []
        cov = None
        if store_covariance:
            cov = np.zeros((n_features, n_features))
        for ind in range(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            means.append(meang)
            # centered group data
            Xgc = Xg - meang
            Xc.append(Xgc)
            if store_covariance:
                cov += np.dot(Xgc.T, Xgc)
        if store_covariance:
            cov /= (n_samples - n_classes)
            self.covariance_ = cov

        self.means_ = np.asarray(means)

        # Transform Y into labels matrix
        lb = preprocessing.LabelBinarizer()
        Y = lb.fit_transform(y)

        # Overall mean
        xbar = np.dot(self.priors_, self.means_)

        # Enter into SDA function
        b = mlab.slda(X, Y, tol=tol, Q=self.n_components)
        self.scalings_ = b
        self.xbar_ = xbar
        # weight vectors / centroids
        self.coef_ = np.dot(self.means_ - self.xbar_, self.scalings_)
        self.intercept_ = (-0.5 * np.sum(self.coef_ ** 2, axis=1) +
                           np.log(self.priors_))