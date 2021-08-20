import logging

from autogluon.core.dataset import TabularDataset
from autogluon.core.features.feature_metadata import FeatureMetadata

try:
    from .version import __version__
except ImportError:
    pass

from .prediction import ImagePredictions

logging.basicConfig(format='%(message)s')  # just print message in logs
