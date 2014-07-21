""" Provides an implementation of Sparse Discriminant Analysis.

This implementation is taken from SpaSM, found at
http://www2.imm.dtu.dk/projects/spasm/ and re-written in Python.

"""

from sklearn import lda, preprocessing, datasets
from sklearn.utils import check_arrays, column_or_1d
import numpy as np
import random
import math


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


def slda(X, Y, delta=1e-6, stop=None, Q=None, maxSteps=100,
         convergenceCriterion=1e-6, verbose=False):

    n, p = X.shape  # n observations, p variables
    K = Y.shape[1]  # K is the number of classes

    if Q is None:
        Q = K - 1  # Q is the number of components
    if Q > K - 1:
        Q = K - 1
        print 'At most K-1 components allowed. Forcing Q = K-1.'

    if stop is None:
        stop = math.ceil(p/2)

    # Setup
    dpi = np.sum(Y, 0) / float(n)  # diagonal "matrix" of class priors
    # class belongings scaled according to priors
    Ydpi = Y / (np.ones((n, 1)) * dpi)
    B = np.zeros((p, Q))  # coefficients of discriminative directions
    theta = np.eye(K, Q)  # optimal scores

    # Main loop
    for q in range(Q):
        step = 0  # iteration counter
        converged = False

        if verbose:
            print 'Estimating direction %s' % q

        while not converged and step < maxSteps:
            step += 1
            Bq_old = B[:, q]

            # 1. Estimate B for the jth direction
            B[:, q] = larsen(X, (Y*theta[:, q])[:, q],
                             delta, stop, [], False, False)
            yhatq = X * B[:, q]

            # 2. Estimate theta
            t = Ydpi.T * yhatq
            s = t - theta[:, :-1] * (theta[:, :-1].T * (np.diag(dpi) * t))
            theta[:, q] = s / np.sqrt(np.sum(dpi.T * s ** 2))

            # converged?
            criterion = np.sum((Bq_old - B[:, q]) ** 2) / (Bq_old.T * Bq_old)
            if verbose and not math.mod(step, 10):
                print 'Iteration: %s, convergence criterion: ' % (
                    step, criterion)
            converged = criterion < convergenceCriterion

            if step == maxSteps:
                print 'Forced exit. Maximum number of steps reached.'

        if verbose:
            print 'Iteration: %s, convergence criterion: %s' % (
                step, criterion)

    return B, theta


def larsen(X, y, delta, stop, Gram, storepath, verbose):
    # algorithm setup
    n, p = X.shape

    # Determine maximum number of active variables
    if delta <= 0:
        maxVariables = min(n, p)  # LASSO
    else:
        maxVariables = p  # Elastic net

    maxSteps = 8 * maxVariables  # Maximum number of algorithm steps

    # set up the LASSO coefficient vector
    if storepath:
        b = np.zeros((p, 2*p))
    else:
        b = np.zeros((p, 1))
        b_prev = b

    # current "position" as LARS travels towards lsq solution
    mu = np.zeros(n)

    # Is a precomputed Gram matrix supplied?
    useGram = Gram is None

    I = range(p)  # inactive set
    A = []  # active set
    if not useGram:
        R = []  # Cholesky factorization R'R = X'X where R is upper triangular

    # correction of stopping criterion to fit naive Elastic Net
    if delta > 0 and stop > 0:
        stop = stop / (1 + delta)

    lassoCond = False  # LASSO condition boolean
    stopCond = False  # Early stopping condition boolean
    step = 1  # step count

    if verbose:
        print 'Step\tAdded\tDropped\t\tActive set size'

    ## LARS main loop
    # while not at OLS solution, early stopping criterion is met, or too many
    # step shave passed
    while len(A) < maxVariables and not stopCond and step < maxSteps:
        print 'DIMS'
        print X.shape
        print y.shape
        print mu.shape
        r = y - mu
        print r.shape

        # find max correlation
        c = X.T * r
        cabs = np.abs(c[I])
        cmax = np.max(cabs)
        cidxI = np.argmax(cabs)
        cidx = I[cidxI]  # index of next active variable

        if not lassoCond:
            # add variable
            if not useGram:
                R = cholinsert(R, X[:, cidx], X[:, A], delta)
            if verbose:
                print '%d\t\t%d\t\t\t\t\t%d\n' % (step, cidx, len(A) + 1)
            A.append(cidx)  # add to active set
            I[cidxI] = []   # ...and drop from inactive set
        else:
            # if a variable has been dropped, do one step with this
            # configuration (don't add new one right away)
            lassoCond = False

        # partial OLS solution and direction from current position to the OLS
        # solution of X_A
        if useGram:
            b_OLS = np.linalg.solve(Gram[A, A], X[:, A].T * y)
        else:
            b_OLS = np.linalg.solve(R, np.linalg.solve(R.T, X[:, A].T * y))
        d = X[:, A] * b_OLS - mu

        # compute length of walk along equiangular direction
        if storepath:
            gamma_tilde = b[A[:-1], step] / (b[A[:-1], step] - b_OLS[:-1, 0])
        else:
            gamma_tilde = b[A[:-1]] / (b[A[:-1]] - b_OLS[:-1, 0])
        gamma_tilde[gamma_tilde <= 0] = np.inf
        dropIdx = np.argmin(gamma_tilde)
        gamma_tilde = np.min(gamma_tilde)

        if len(I) == 0:
            # if all variables active, go all the way to the OLS solution
            gamma = 1
        else:
            cd = X.T * d
            temp = np.array([(c[I] - cmax) / (cd[I] - cmax),
                             (c[I] + cmax) / (cd[I] + cmax)])
            temp = temp[temp > 0].sort()
            if len(temp) == 0:
                print ('Could not find a positive direction'
                       'towards the next event')
                return
            gamma = temp[0]

        # check if variable should be dropped
        if gamma_tilde < gamma:
            lassoCond = 1
            gamma = gamma_tilde

        # update beta
        if storepath:
            # check if beta must grow
            if b.shape[1] < step + 1:
                b.append(np.zeros((p, b.shape[1])))
            # update beta
            b[A, step+1] = b[A, step] + gamma * (b_OLS - b[A, step])
        else:
            b_prev = b
            b[A] += gamma * (b_OLS - b[A])

        # update position
        mu += gamma * d

        # increment step counter
        step += 1

        # Early stopping at specified bound on L1 norm of beta
        if stop > 0:
            if storepath:
                t2 = np.sum(np.abs(b[:, step]))
                if t2 >= stop:
                    t1 = np.sum(np.abs(b[:, step-1]))
                    s = (stop - t1) / (t2 - t1)  # interpolation factor 0<s<1
                    b[:, step] = b[:, step-1] + s*(b[:, step] - b[:, step-1])
                    stopCond = True
            else:
                t2 = np.sum(np.abs(b))
                if t2 >= stop:
                    t1 = np.sum(np.abs(b_prev))
                    s = (stop - t1) / (t2 - t1)  # interpolation factor 0<s<1
                    b = b_prev + s * (b - b_prev)
                    stopCond = True

        # If LASSO condition satisfied, drop variable from active set
        if lassoCond:
            if verbose:
                print '%d\t\t\t\t%d\t\t\t%d\n' % (step, A[dropIdx], len(A)-1)
            if not useGram:
                R = choldelete(R, dropIdx)
            I.append(A[dropIdx])  # add dropped variable to inactive set
            A[dropIdx] = []  # ...and remove from active set

        # Early stopping at specified number of variables
        if stop < 0:
            stopCond = len(A) >= -stop

    # trim beta
    if storepath and b.shape[1] > step:
        b[:, step+1:] = []

    # return number of iterations
    steps = step - 1

    # issue warning if algorithm did not converge
    if step == maxSteps:
        print 'Forced exit. Maximum number of steps reached.'

    return b, steps


iris = datasets.load_iris()
X = iris.data
y = iris.target
lb = preprocessing.LabelBinarizer()
Y = lb.fit_transform(y)

print X.shape
print Y.shape
slda(X, Y)
