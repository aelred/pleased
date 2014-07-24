""" Provides an implementation of Sparse Discriminant Analysis.

This implementation is taken from SpaSM, found at
http://www2.imm.dtu.dk/projects/spasm/ and re-written in Python.

"""

from sklearn import preprocessing
from sklearn.utils import check_arrays, column_or_1d
import numpy as np
import pyper


class SDA:
    """
    Sparse Discriminant Analysis (SDA)

    A classifier that behaves like LDA while maintaining a
    sparse coefficient matrix.
    In other words, the number of features used is minimized.
    """

    def __init__(self, n_components=None, num_features=None):
        self.n_components = n_components
        self.num_features = num_features
        self.r = pyper.R()
        self.r('library(sparseLDA)')

    def fit(self, X, y, tol=1.0e-6):
        X, y = check_arrays(X, y, sparse_format='dense')
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        n_components = self.n_components or n_classes-1
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')

        # Transform y into labels matrix Y
        lb = preprocessing.LabelBinarizer()
        Y = lb.fit_transform(y)

        # Enter into SDA function
        self.r['X'] = np.matrix(X)
        self.r['Y'] = np.matrix(Y)
        self.r['colnames(Y)'] = self.classes_
        self.r['tol'] = tol
        self.r['Q'] = n_components
        if self.num_features is None:
            self.r['stop'] = -n_features / 2
        else:
            self.r['stop'] = -self.num_features
        print self.r('out <- sda(X, Y, tol=tol, Q=Q, stop=stop, trace=TRUE)')

        b = self.r['out']['beta']
        v = self.r['out']['varIndex']
        self.scalings_ = np.zeros((n_features, n_components))
        for i, scales in zip(v, b):
            self.scalings_[i] = scales

    def transform(self, X):
        self.r['X'] = np.matrix(X)
        self.r('p <- predict(out, X)')
        return self.r['p']['x']

    def predict(self, X):
        self.r['X'] = np.matrix(X)
        self.r('p <- predict(out, X)')
        return self.r['p']['class']
