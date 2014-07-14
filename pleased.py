from learn import *
from transform import *

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
