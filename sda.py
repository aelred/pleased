""" Provides an implementation of Sparse Discriminant Analysis. """

from sklearn import lda, preprocessing, utils
import numpy as np
import random


class SDA(lda.LDA):
    """
    Sparse Discriminant Analysis (SDA)

    A classifier that behaves like LDA while maintaining a 
    sparse coefficient matrix.
    In other words, the number of features used is minimized.
    """

    def fit(self, X, y):
        X, y = utils.check_arrays(X, y, sparse_format='dense')
        y = utils.column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)

        n, p = X.shape
        K = len(self.classes_)

        if K < 2:
            raise ValueError('y has less than 2 classes')
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(K)
        else:
            self.priors_ = self.priors

        # 1. Let Y be a n x K matrix of indicator variables
        y_dict = dict(enumerate(y))
        lb = preprocessing.LabelBinarizer()
        lb.fit(y)
        Y = lb.fit_transform(y)

        # 2. Let D = (1/n)Y'Y
        D = (1 / n) * Y.T * Y

        # 3. Let Q[0] be a K x 1 matrix of 1's.
        Q[0] = np.ones((K, 1))

        # 4. For k = 0,....,q, compute a new SDA direction pair (theta[k], beta[k])
        for k = range(q):
            # (a) Initialize theta[k] = (I - QkQk'D)theta_s, where theta_s is a 
            #     random K-vector, and then normalize theta[k] so that 
            #     theta[k]' * D * theta[k] = 1.
            theta_s = np.array([random.uniform(-1.0, 1.0) for i in range(K)])
            theta[k] = (np.identity() - Q[k] * Q[k].T * D) - theta_s
            theta[k] *= 1.0 / (theta[k].T * D * theta[k])**0.5

            # ...