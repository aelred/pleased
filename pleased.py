from learn import *
from transform import *
from sda import SDA

# bare minimum preprocessing to give valid data
preproc_min = [
    ('avg', ElectrodeAvgTransform()),
    ('poststim', PostStimulusTransform())
]

# averages electrodes and detrends data
preproc_standard = [
    ('avg', ElectrodeAvgTransform()),
    ('detrend', DetrendTransform()),
    ('poststim', PostStimulusTransform()),
]

ensemble = FeatureEnsembleTransform().extractor
window = WindowTransform(ensemble, 3, False).extractor

# applies feature ensemble to decimated windows
extract_decimate_ensemble = [
    ('feature', DecimateWindowTransform(window)),
]

# normalizes data
postproc_standard = [
    ('scaler', preprocessing.StandardScaler())
]

# classifier that extracts features from decimated windows
feat_class = Classifier(
    preproc_standard, 
    extract_decimate_ensemble, 
    postproc_standard, 
    svm.SVC())

min_class = Classifier(preproc_min, [], postproc_standard, svm.SVC())

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

null_class = NullClassifier(preproc_min, [], postproc_standard, svm.SVC())


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
    detrend_class = Classifier(preproc_standard, [], postproc_standard, svm.SVC())
    detrend_null = NullClassifier(preproc_standard, [], 
                                  postproc_standard, svm.SVC())
    detrend_class.plot('Separation with linear detrending')
    detrend_null.plot3d('Separation of null data with linear detrending', False)


def basic_features():
    """
    2014-07-14
    Plot separation of basic feature extraction methods based on the mean.
    """
    def extract(x):
        return [mean(x), mean(map(abs, differential(x))), 
                mean(map(abs, differential(differential(x))))]
    feature_class = Classifier(preproc_standard, 
                               [('features', Extractor(extract))], 
                               postproc_standard, svm.SVC())
    feature_class.plot('Separation using basic features')
    feature_class.plot_lda_scaling(True, 'Significance of basic features',
        ['mean', 'diff1', 'diff2'])


def basic_features2():
    """
    2014-07-14
    Plot separation using another set of basic features based on variance.
    """
    def extract(x):
        return [var(x), var(differential(x)), 
                var(differential(differential(x)))]
    feature_class = Classifier(preproc_standard, 
                               [('features', Extractor(extract))], 
                               postproc_standard, svm.SVC())
    feature_class.plot('Separation using basic features')
    feature_class.plot_lda_scaling(True, 'Significance of basic features',
        ['var', 'var(diff1)', 'var(diff2)'])


def feature_ensemble():
    """
    2014-07-15
    Plot separation using many features.
    """
    feature_class = Classifier(preproc_standard, 
                               [('features', FeatureEnsembleTransform())], 
                               postproc_standard, svm.SVC())
    feature_class.plot('Separation using multiple time-series features')
    feature_class.plot_lda_scaling(True, 'Significance of time-series features',
        ['mean', 'mean(diff1)', 'mean(diff2)', 'var', 'var(diff1)', 'var(diff2)',
         'hmob', 'hcom', 'skewness', 'kurtosis'])


def noise_extraction():
    """
    2014-07-15
    Plot separation using the noise of the signal.
    """
    classifier = Classifier(preproc_min, 
                            [('noise', NoiseTransform(100))], 
                            postproc_standard, svm.SVC())
    classifier.plot('Separation using noise in time-series')
    classifier.plot_lda_scaling(False, 'Significance of noise in time-series')


def noise_features():
    """
    2014-07-15
    Plot separation using the noise of the signal and the feature ensemble.
    """
    classifier = Classifier(preproc_min, 
                            [('noise', NoiseTransform(100)), 
                             ('features', FeatureEnsembleTransform())], 
                            postproc_standard, svm.SVC())
    classifier.plot('Separation using noise and feature ensemble')
    classifier.plot_lda_scaling(True, 'Significance of features in noise.',
        ['mean', 'mean(diff1)', 'mean(diff2)', 'var', 'var(diff1)', 'var(diff2)',
         'hmob', 'hcom', 'skewness', 'kurtosis'])


def separate_electrodes():
    """
    2014-07-16
    Plot separation when operations are performed on each electrode separately.
    """

    # detrend each electrode individually
    preproc_separate = [
        ('detrend', MapElectrodeTransform(DetrendTransform().extractor)),
        ('poststim', MapElectrodeTransform(PostStimulusTransform().extractor)),
    ]

    features_separate = [
        ('features', MapElectrodeTransform(FeatureEnsembleTransform().extractor))
    ]

    # concatenate results together
    postproc_separate = [
        ('concat', ConcatTransform())
    ] + postproc_standard

    classifier = Classifier(preproc_separate, features_separate, 
                            postproc_separate, svm.SVC())
    classifier.plot('Separation using both electrode readings')


def fourier_feature():
    """
    2014-07-16
    Plot separation using a Fourier transform of the signal.
    """

    features = [('noise', NoiseTransform(100)), ('fourier', FourierTransform())]

    classifier = Classifier(preproc_standard, features, 
                            postproc_standard, svm.SVC())
    classifier.plot('Separation using a Fourier transform')


def sda_separation():
    """
    2014-07-17
    Plot separation using Sparse Discriminant Analysis.
    """

    classifier = Classifier(preproc_min, [], postproc_standard, svm.SVC(), SDA())
    classifier.plot('Separation using SDA')
    classifier.plot_lda_scaling(False, 'Significance of features using SDA scaling')