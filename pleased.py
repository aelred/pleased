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

_ensemble = FeatureEnsembleTransform().extractor
_window = WindowTransform(_ensemble, 3, False).extractor

# applies feature ensemble to decimated windows
extract_decimate_ensemble = [
    ('feature', DecimateWindowTransform(_window)),
]

# normalizes data
postproc_standard = [
    ('scaler', preprocessing.StandardScaler())
]

# classifier that performs the bare minimum transforms on the raw data
min_class = Classifier(preproc_min, [], postproc_standard, svm.SVC())

# classifier that extracts features from decimated windows
feat_class = Classifier(
    preproc_standard, 
    extract_decimate_ensemble, 
    postproc_standard, 
    svm.SVC())


null_class = NullClassifier(preproc_min, [], postproc_standard, svm.SVC())

def null_only_classify():
    """
    2014-07-11:
    Plot classification of null data by experiment.
    """
    null_class.labels = ['null']
    null_class.plot()


def null_all_classify():
    """ 
    2014-07-11:
    Plot classification of null data by experiment as well as non-null data. 
    """
    null_class.labels = learn.def_labels
    null_class.plot()
