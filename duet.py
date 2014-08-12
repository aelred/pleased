""" Degenerate Unmixing Estimation Technique for blind source separation. """

import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import transform


class DUET:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print "GO"
        return duet(X.T[0], X.T[1], *self.args, **self.kwargs).T

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        self.transform(X)
        return self


def duet(x1, x2, numsources=2, numfreq=4096, wlen=4096,
         timestep=512, p=0.5, q=0, bin=4.0):
    ## 1. analyze the signals - STFT
    print "STFT"
    awin = scipy.hamming(wlen)  # analysis window is a Hamming window

    time_freq = lambda x: tfanalysis(x, awin, timestep, numfreq)
    # time-freq domain
    tf1, tf2 = time_freq(x1), time_freq(x2)
    # remove dc component from mixtures to avoid dividing by zero
    # frequency in the delay estimation
    tf1 = tf1[1:, :]
    tf2 = tf2[1:, :]

    # calculate pos/neg frequencies for later use in delay calc
    freq = np.append(np.arange(1, numfreq / 2+1),
                     np.arange((-numfreq / 2)+1, 0)) * (2 * np.pi / numfreq)
    freq = freq.reshape((1, -1))
    fmat = np.tile(freq, (tf1.shape[1], 1)).T

    ## 2. calculate alpha and delta for each t-f point
    R21 = (tf2 + sys.float_info.epsilon) / (tf1 + sys.float_info.epsilon)

    ## 2.1 HERE WE ESTIMATE THE RELATIVE ATTENUATION (alpha)
    a = np.abs(R21)
    alpha = a - 1.0 / a
    ## 2.2 HERE WE ESTIMATE THE RELATIVE DELAY (delta)
    delta = -np.imag(np.log(R21)) / fmat

    ## 3. calculate weighted histogram
    print "Histogram"
    tfweight = (np.abs(tf1) * np.abs(tf2))**p * np.abs(fmat)**q
    maxa = 0.7
    maxd = 3.6
    abins = 35 * bin
    dbins = 50 * bin

    # only consider time-freq points yielding estimates in bounds
    amask = (np.logical_and(np.abs(alpha) < maxa, np.abs(delta) < maxd))
    alpha_vec = alpha[amask]
    delta_vec = delta[amask]
    tfweight = tfweight[amask]
    # determine histogram indices
    alpha_ind = np.round((abins-1) * (alpha_vec+maxa) / (2*maxa))
    delta_ind = np.round((dbins-1) * (delta_vec+maxd) / (2*maxd))

    A, edges = np.histogramdd(np.array((alpha_ind, delta_ind)).T,
                              bins=(abins, dbins), weights=tfweight)
    # smooth the histogram
    A = smooth2d(A, 3)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # lina = np.linspace(-maxa, maxa, abins)
    # lind = np.linspace(-maxd, maxd, dbins)
    # lind, lina = np.meshgrid(lind, lina)
    # ax.plot_surface(lina, lind, A, cmap=cm.coolwarm)
    # ax.set_xlabel('symmetric attenuation (alpha)')
    # ax.set_ylabel('relative delay (delta)')
    # plt.show()

    ## 4. peak centers (determined from histogram)
    print "Peak-finding"
    peakdelta = [-2, -2, 0, 2, 2]
    peakalpha = [0.19, -0.21, 0, 0.19, -0.21]

    # covert alpha to a
    peaka = (peakalpha+np.sqrt(np.power(peakalpha, 2) + 4)) / 2.0

    ## 5. determine masks for separation
    print "Mask"
    bestsofar = np.inf * np.ones(tf1.shape)
    bestind = np.zeros(tf1.shape)
    im = -np.sqrt(-1+0j)
    for i, a in enumerate(peakalpha):
        score = np.abs(peaka[i] * np.exp(-im * fmat * peakdelta[i]) * tf1 - tf2)
        score = score ** 2 / (1 + peaka[i] ** 2)
        mask = (score < bestsofar)
        bestind[mask] = i
        bestsofar[mask] = score[mask]

    ## 6. & 7. demix with ML alignment and convert to time domain
    print "Demix"
    est = np.zeros((numsources, len(x1)))
    for i in range(numsources):
        mask = (bestind == i)
        r1 = np.zeros((1, tf1.shape[1]))
        r2 = (((tf1 + peaka[i] * np.exp(im * fmat * peakdelta[i]) * tf2)
               / (1 + peaka[i] ** 2)) * mask)
        esti = tfsynthesis(np.concatenate([r1, r2], 0), np.sqrt(2) * awin / 1024,
                           timestep, numfreq)
        est[i, :] = esti[:len(x1), 0]

    # plt.plot(est.T)
    # plt.show()

    return est


def smooth2d(mat, ker):
    try:
        ker[0]
    except TypeError:
        kmat = np.ones((ker, ker)) / ker**2
    else:
        kmat = ker

    # make kmat have odd dimensions
    kr, kc = kmat.shape
    if kr % 2 == 0:
        kmat = scipy.signal.convolve2d(kmat, np.ones((1, 2))) / 2
        kc += 1

    mr, mc = mat.shape
    mr -= 1
    mc -= 1
    fkr = np.floor(kr / 2)  # number of rows to copy on top and bottom
    fkc = np.floor(kc / 2)  # number of columns to copy on either side

    r1 = np.concatenate([mat[0, 0] * np.ones((fkr, fkc)),
                         np.ones((fkr, 1)) * mat[0, :],
                         mat[0, mc] * np.ones((fkr, fkc))], 1)
    r2 = np.concatenate([(mat[:, 0] * np.ones((1, fkc))).T, mat,
                         (mat[:, mc] * np.ones((1, fkc))).T], 1)
    r3 = np.concatenate([mat[mr, 0] * np.ones((fkr, fkc)),
                         np.ones((fkr, 1)) * mat[mr, :],
                         mat[mr, mc] * np.ones((fkr, fkc))], 1)
    r = np.concatenate([r1, r2, r3])
    smat = scipy.signal.convolve2d(r, np.flipud(np.fliplr(kmat)), mode='valid')
    return smat


def tfsynthesis(timefreqmat, swin, timestep, numfreq):
    winlen = len(swin)
    numfreq, numtime = timefreqmat.shape
    ind = np.remainder(np.arange(0, winlen) - 1, numfreq)
    x = np.zeros((numtime * timestep + winlen, 1))

    for i in range(numtime):
        temp = numfreq * np.real(scipy.ifft(timefreqmat[:, i]))
        sind = (i-1) * timestep
        rind = np.arange(sind, sind+winlen)
        x[rind] += (temp[ind] * swin).reshape(-1, 1)

    return x


def tfanalysis(x, awin, timestep, numfreq):
    nsamp = len(x)
    wlen = len(awin)

    # calc size and init output t-f matrix
    numtime = int(np.ceil((nsamp-wlen+1.0) / timestep))
    tfmat = np.zeros((numfreq, numtime+1), dtype=np.complex_)

    i = 0
    for i in range(0, numtime):
        sind = i*timestep
        tfmat[:, i] = scipy.fft(x[sind:sind+wlen] * awin, numfreq)
    i += 1
    sind = i*timestep
    lasts = min(sind, len(x))
    laste = min((sind+wlen), len(x))
    pad = np.zeros(wlen-(laste-lasts))
    tfmat[:, -1] = scipy.fft(np.append(x[lasts:laste], pad) * awin, numfreq)
    return tfmat
