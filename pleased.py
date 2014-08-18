from learn import *
from transform import *
from sda import SDA
from sklearn import preprocessing
from itertools import chain
import scipy
import random

dec = [('dec', Decimate(16))]

# bare minimum preprocessing to give valid data
preproc_min = [
    ('avg', ElectrodeAvg()),
    ('poststim', PostStimulus())
] + dec

# averages electrodes and detrends data
preproc_standard = [
    ('avg', ElectrodeAvg()),
    ('detrend', Detrend()),
    ('poststim', PostStimulus()),
]

# peprocess and decimate window (reduce size by factor of 10)
preproc_dec = preproc_standard + dec

# detrend each electrode separately
preproc_separate = [
    ('concat', Concat()),
    ('detrend', Map(Detrend(), divs=2)),
    ('poststim', Map(PostStimulus(), divs=2)),
]

ensemble = FeatureEnsemble()
window = Window(ensemble, 3, False)

# applies feature ensemble to decimated windows
extract_decimate_ensemble = [
    ('feature', DecimateWindow(window)),
]

# normalizes data
postproc_standard = [
    ('scaler', preprocessing.StandardScaler())
]

# classifier that extracts features from decimated windows
feat_class = Classifier(
    preproc_standard,
    extract_decimate_ensemble,
    postproc_standard)

min_class = Classifier(preproc_min, [], postproc_standard)


def basic_separator():
    """
    2014-07-11
    Plot separation of labels with minimal pre-processing of the data.
    """
    min_class.plot('Separation with minimal pre-processing.', False)


def basic_separator_validation():
    """
    2014-07-11
    Test basic separator using separate training and validation sets.
    """
    min_class.plot('Separation with minimal pre-processing.', True)

null_class = NullClassifier(preproc_min, [], postproc_standard)


def null_only_plot():
    """
    2014-07-11
    Plot separation of null data by experiment.
    """
    null_class.labels = ['null']
    null_class.plot3d('Separation of null data by experiment type', False)


def null_all_plot():
    """
    2014-07-11
    Plot separation of null data by experiment as well as non-null data.
    """
    null_class.labels = def_labels
    null_class.plot3d('Separation of null data by experiment type '
                      'and stimuli by stimulus type', False)


def basic_separator_features():
    """
    2014-07-14
    Plot which features are important for the basic LDA separator.
    """
    min_class.plot_lda_scaling(False, 'Significance of time series features')


def linear_detrending():
    """
    2014-07-14
    Plot separation with linear detrending applied to remove experimental bias.
    """
    detrend_class = Classifier(preproc_dec, [], postproc_standard)
    detrend_null = NullClassifier(preproc_dec, [], postproc_standard)
    detrend_class.plot('Separation with linear detrending')
    detrend_null.plot3d('Separation of null data with linear detrending', False)


def basic_features():
    """
    2014-07-14
    Plot separation of basic feature extraction methods based on the mean.
    """
    m = Mean()
    a = Abs()
    d = Differential()

    def extract(x):
        return [m(x), m(a(d(x))), m(a(d(d(x))))]
    feature_class = Classifier(preproc_dec,
                               [('features', Extractor(extract))],
                               postproc_standard)
    feature_class.plot('Separation using basic features')
    feature_class.plot_lda_scaling(True, 'Significance of basic features',
                                   ['mean', 'diff1', 'diff2'])


def basic_features2():
    """
    2014-07-14
    Plot separation using another set of basic features based on variance.
    """
    v = Var()
    d = Differential()

    def extract(x):
        return [v(x), v(d(x)), v(d(d(x)))]
    feature_class = Classifier(preproc_dec,
                               [('features', Extractor(extract))],
                               postproc_standard)
    feature_class.plot('Separation using basic features')
    feature_class.plot_lda_scaling(True, 'Significance of basic features',
                                   ['var', 'var(diff1)', 'var(diff2)'])


def feature_ensemble():
    """
    2014-07-15
    Plot separation using many features.
    """
    feature_class = Classifier(preproc_dec,
                               [('features', FeatureEnsemble())],
                               postproc_standard)
    feature_class.plot('Separation using multiple time-series features')
    feature_class.plot_lda_scaling(True, 'Significance of time-series features',
                                   ['mean', 'mean(diff1)', 'mean(diff2)', 'var',
                                    'var(diff1)', 'var(diff2)',
                                    'hmob', 'hcom', 'skewness', 'kurtosis'])


def noise_extraction():
    """
    2014-07-15
    Plot separation using the noise of the signal.
    """
    classifier = Classifier(preproc_min,
                            [('noise', Noise(100))],
                            postproc_standard)
    classifier.plot('Separation using noise in time-series')
    classifier.plot_lda_scaling(False, 'Significance of noise in time-series')


def noise_features():
    """
    2014-07-15
    Plot separation using the noise of the signal and the feature ensemble.
    """
    classifier = Classifier(preproc_min,
                            [('noise', Noise(100)),
                             ('features', FeatureEnsemble())],
                            postproc_standard)
    classifier.plot('Separation using noise and feature ensemble')
    classifier.plot_lda_scaling(True, 'Significance of features in noise.',
                                ['mean', 'mean(diff1)', 'mean(diff2)',
                                 'var', 'var(diff1)', 'var(diff2)',
                                 'hmob', 'hcom', 'skewness', 'kurtosis'])


def separate_electrodes():
    """
    2014-07-16
    Plot separation when operations are performed on each electrode separately.
    """
    features_separate = [
        ('features', Map(FeatureEnsemble(), divs=2))
    ]

    classifier = Classifier(preproc_separate + dec, features_separate,
                            postproc_standard)
    classifier.plot('Separation using both electrode readings')
    classifier.plot_lda_scaling(True,
                                'Significance of features across both electrodes.',
                                ['mean A', 'mean(diff1) A', 'mean(diff2) A',
                                 'var A', 'var(diff1) A', 'var(diff2) A',
                                 'hmob A', 'hcom A', 'skewness A', 'kurtosis A',
                                 'mean B', 'mean(diff1) B', 'mean(diff2) B',
                                 'var B', 'var(diff1) B', 'var(diff2) B',
                                 'hmob B', 'hcom B', 'skewness B', 'kurtosis B'])


def fourier_feature():
    """
    2014-07-16
    Plot separation using a Fourier transform of the signal.
    """

    features = [('noise', Noise(100)), ('fourier', Fourier())]

    classifier = Classifier(preproc_dec, features, postproc_standard)
    classifier.plot('Separation using a Fourier transform')


def sda_separation():
    """
    2014-07-17
    Plot separation using Sparse Discriminant Analysis.
    """

    classifier = Classifier(preproc_min, [], postproc_standard, SDA())
    classifier.plot('Separation using SDA')
    classifier.plot_lda_scaling(False, 'Significance of features using SDA scaling')


def sda_separation_50():
    """
    2014-07-24
    Plot separation using SDA and only a small number of features.
    """

    classifier = Classifier(preproc_min, [], postproc_standard,
                            svm.SVC(), SDA(num_features=50))
    classifier.plot('Separation using SDA and 50 features.')
    classifier.plot_lda_scaling(False, 'Significance of features using SDA scaling')


def wavelet_separation():
    """
    2014-07-24
    Plot separation using SDA on a wavelet transform.
    """

    features = [('wavelet', DiscreteWavelet('haar', 15, 0, True))]
    classifier = Classifier(preproc_standard, features, postproc_standard,
                            svm.SVC())  # SDA(num_features=50))
    classifier.plot('Separation using SDA on wavelet transform.')
    classifier.plot_lda_scaling(False, 'Signifiance of wavelet transform features.')


def wavelet_feature():
    """
    2014-07-28
    Plot separation using SDA on feature ensemble of the wavelet transform.
    """
    num_levels = 15
    drop_levels = 3
    ensembles = [FeatureEnsemble() for i in range(num_levels-drop_levels)]
    features = [
        ('wavelet',
         DiscreteWavelet('haar', num_levels, drop_levels, True, ensembles))
    ]
    classifier = Classifier(preproc_standard, features,
                            postproc_standard, SDA(num_features=20))
    classifier.plot('Separation using feature ensemble on wavelet transform.')
    labels = list(chain(
        *[[i, '', '', 'v', '', '', 'h', '', '', '']
          for i in range(num_levels-drop_levels)]))
    classifier.plot_lda_scaling(
        True, 'Significance of wavelet transform features.', labels)


def cross_correlation():
    """
    2014-07-30
    Plot separation using cross-correlation of electrode channels.
    """

    mov_avg = Map(MovingAvg(100), divs=2)
    deriv = Map(Differential(), divs=2)
    mean = Map(MeanSubtract(), divs=2)
    features = [('m', mov_avg), ('d', deriv), ('me', mean),
                ('a', Abs()), ('cr', CrossCorrelation())]
    classifier = Classifier(preproc_separate + dec, features,
                            postproc_standard, SDA(num_features=50))
    classifier.plot('Separation using cross-correlation of electrode channels.')
    classifier.plot_lda_scaling(False, 'Significance of cross-correlation values.')


def cross_correlation_windowed():
    """
    2014-07-30
    Plot separation using cross-correlation of electrode channels.
    """

    mov_avg = Map(MovingAvg(100), divs=2)
    deriv = Map(Differential(), divs=2)
    mean = Map(MeanSubtract(), divs=2)
    window = Extractor(lambda x: x * np.hanning(len(x)))
    features = [('m', mov_avg), ('d', deriv), ('me', mean),
                ('a', Abs()), ('cr', CrossCorrelation()), ('w', window)]
    classifier = Classifier(preproc_separate + dec, features,
                            postproc_standard, SDA(num_features=50))
    classifier.plot('Separation using cross-correlation of electrode channels.')
    classifier.plot_lda_scaling(False, 'Significance of cross-correlation values.')


def time_delay():
    """
    2014-07-30
    Plot separation by calculating the time delay between electrode channels.
    """

    mov_avg = Map(MovingAvg(100), divs=2)
    deriv = Map(Differential(), divs=2)
    mean = Map(MeanSubtract(), divs=2)
    features = [('m', mov_avg), ('d', deriv), ('me', mean),
                ('a', Abs()), ('t', TimeDelay())]
    classifier = Classifier(preproc_separate + dec, features,
                            postproc_standard, lda.LDA())
    classifier.plot1d('Separation using time delay between electrode channels.')


def cross_correlation_ensemble():
    """
    2014-07-30
    Plot separation by calculating the feature ensemble on cross-correlation data.
    """

    mov_avg = Map(MovingAvg(100), divs=2)
    deriv = Map(Differential(), divs=2)
    mean = Map(MeanSubtract(), divs=2)
    features = [('m', mov_avg), ('d', deriv), ('me', mean),
                ('a', Abs()), ('cr', CrossCorrelation()), ('f', FeatureEnsemble())]
    classifier = Classifier(preproc_separate + dec, features,
                            postproc_standard, lda.LDA())
    classifier.plot('Separation using features of cross-correlation.')
    classifier.plot_lda_scaling(True, 'Significance of cross-correlation features.',
                                ['mean', 'mean(diff1)', 'mean(diff2)',
                                 'var', 'var(diff1)', 'var(diff2)',
                                 'hmob', 'hcom', 'skewness', 'kurtosis'])


def multiple_ensembles():
    """
    2014-07-30
    Plot separation using combinations of feature ensembles from:
        1. The averaged electrode data
        2. The noise (subtracting a moving average)
        3. The wavelet transform
        4. The cross-correlation
    """

    pre = [('concat', Concat()),
           ('detrend', Map(Detrend(), divs=2)),
           ('post', Map(PostStimulus(), divs=2))]

    feature = ('feature', FeatureEnsemble())
    avg = ('avg', ElectrodeAvg())

    avg_feat = pipeline.Pipeline([avg] + dec + [feature])
    noise = pipeline.Pipeline([avg] + dec + [('noise', Noise(100)), feature])

    num_levels = 15
    drop_levels = 3
    wavelet = pipeline.Pipeline(
        [avg,
         ('wavelet',
          DiscreteWavelet('haar', num_levels, drop_levels, True,
                          [FeatureEnsemble()] * (num_levels - drop_levels)))
         ])

    mov_avg = Map(MovingAvg(100), divs=2)
    deriv = Map(Differential(), divs=2)
    mean = Map(MeanSubtract(), divs=2)
    cross = pipeline.Pipeline([('m', mov_avg), ('d', deriv), ('me', mean),
                               ('a', Abs()), ('cr', CrossCorrelation()), feature])

    union = pipeline.FeatureUnion([('a', avg_feat), ('n', noise),
                                   ('w', wavelet), ('c', cross)])
    classifier = Classifier(pre, [('union', union)], postproc_standard,
                            svm.SVC(), SDA(num_features=50))
    classifier.plot('Separation using multiple feature ensembles.')

    lab_f = lambda name: [name, '', '', 'v', '', '', 'h', '', '', '']
    labels = lab_f('a') + lab_f('n') + (
        list(chain(*[lab_f('w%d' % i) for i in range(13)]))) + lab_f('c')

    classifier.plot_lda_scaling(True, 'Significance of multiple feature ensembles.',
                                labels)


def null_separation_validation():
    """
    2014-07-31
    Plot separation of null data using a validation set.
    """
    null_class.labels = def_labels
    null_class.plot3d('Separation of null data by experiment type '
                      'and stimuli by stimulus type')
    null_class.plot_lda_scaling(False, 'Significance of null separation features.')


def wavelet_null_separation():
    """
    2014-07-31
    Plot separation of null data using SDA on a wavelet transform.
    """

    features = [('wavelet', DiscreteWavelet('haar', 11, 0, True))]
    classifier = NullClassifier(preproc_standard, features, postproc_standard,
                                svm.SVC(), SDA(num_features=50))
    classifier.plot3d('Null separation using SDA on wavelet transform.')
    classifier.plot_lda_scaling(False, 'Signifiance of wavelet transform features.')


def noise_correlation_separation():
    """
    2014-08-01
    Plot separation using cross-correlation of the derivative of the noise.
    """
    mov_avg = Map(MovingAvg(100), divs=2)
    deriv = Map(Differential(), divs=2)
    mean = Map(MeanSubtract(), divs=2)
    classifier = Classifier(preproc_separate,
                            [('n', Map(Noise(100), divs=2)),
                             ('m', mov_avg), ('d', deriv), ('me', mean),
                             ('a', Abs()), ('c', CrossCorrelation()),
                             ('f', FeatureEnsemble())],
                            postproc_standard)
    classifier.plot('Separation using cross-correlation of noise derivative.')
    classifier.plot_lda_scaling(True, 'Significance of cross-correlation of noise',
                                ['mean', 'mean(diff1)', 'mean(diff2)',
                                 'var', 'var(diff1)', 'var(diff2)',
                                 'hmob', 'hcom', 'skewness', 'kurtosis'])


def histogram_ben_separation():
    """
    2014-08-01
    Plot separation of Ben's wavelet histogram data from Matlab.
    """
    mat = scipy.io.loadmat('ben.mat')
    X = mat['dwtFeats']
    y = mat['classes'].ravel()
    sources = [''] * len(y)
    classifier = Classifier([], [], postproc_standard)

    cutoff = 0.75 * len(y)
    seed = random.random()
    random.seed(seed)
    random.shuffle(X)
    random.seed(seed)
    random.shuffle(y)
    X_train, y_train = X[:cutoff], y[:cutoff]
    X_valid, y_valid = X[cutoff:], y[cutoff:]

    # put flag inside a list to deal with weird scoping
    train_set = [True]

    # the second time this is called, return a different set of data
    def get_data(_=None):
        if train_set[0]:
            train_set[0] = False
            return (X_train, y_train, sources)
        else:
            return (X_valid, y_valid, sources)
    classifier.get_data = get_data
    classifier.plot1d('Separation using Ben\'s histograms.')
    classifier.plot_lda_scaling(False, 'Significance of histogram features.')


def histogram_my_separation():
    """
    2014-08-01
    Attempt to emulate Ben's results above.
    """
    num_levels = 15
    drop_levels = 3
    histograms = [Histogram(10) for x in range(num_levels-drop_levels)]
    features = [
        ('wavelet',
         DiscreteWavelet('haar', num_levels, drop_levels, True, histograms))
    ]
    classifier = Classifier(preproc_standard, features,
                            postproc_standard)
    # classifier.labels = ['null', 'ozone']
    classifier.plot('Separation using histogram of wavelets.')
    classifier.plot_lda_scaling(False, 'Significance of histogram features.')


def ica_noise_separation():
    """
    2014-08-12
    Separate data using independent component analysis of the noise.
    """
    features = [('noise', Map(Noise(1000), divs=2)),
                ('ica', ICA()),
                ('features', Map(FeatureEnsemble(), divs=2))]
    classifier = Classifier(preproc_separate, features,
                            postproc_standard)
    classifier.plot('Separation using ICA of noise')
    classifier.plot_lda_scaling(True, 'Significance of features in ICA of noise.',
                                ['mean', 'mean(diff1)', 'mean(diff2)',
                                 'var', 'var(diff1)', 'var(diff2)',
                                 'hmob', 'hcom', 'skewness', 'kurtosis'] * 2)


def mult_noise_separation():
    """
    2014-08-13
    Separate data using the high-frequencies of both channels multiplied together.
    Positive values will indicate correlation in the channels, negative values
    indicate anticorrelation.
    """
    mult = ElectrodeOp(lambda x1, x2: x1 * x2)
    features = [('n', Map(Noise(1000), divs=2)),
                ('m', mult), ('f', FeatureEnsemble())]
    classifier = Classifier(preproc_separate, features,
                            postproc_standard, SDA(num_features=50))
    classifier.plot('Separation using multiplied noise.')
    classifier.plot_lda_scaling(True,
                                'Significance of features in multiplied noise.',
                                ['mean', 'mean(diff1)', 'mean(diff2)',
                                 'var', 'var(diff1)', 'var(diff2)',
                                 'hmob', 'hcom', 'skewness', 'kurtosis'])
