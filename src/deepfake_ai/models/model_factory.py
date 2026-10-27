"""Unified model instantiation factory."""

from deepfake_ai.core.exceptions import ModelError
from deepfake_ai.models.deep_learning.mesonet import Meso4, MesoInception4
from deepfake_ai.models.deep_learning.xception import DeepfakeXception
from deepfake_ai.models.deep_learning.efficientnet import DeepfakeEfficientNet
from deepfake_ai.models.deep_learning.frequency_detector import DualStreamFrequencyNet
from deepfake_ai.models.deep_learning.spatiotemporal_detector import SpatioTemporalDetector
from deepfake_ai.models.deep_learning.diffusion_detector import DiffusionResidualDetector
from deepfake_ai.models.classical.svm import build_svm
from deepfake_ai.models.classical.random_forest import build_random_forest
from deepfake_ai.models.classical.gradient_boosting import build_gradient_boosting
from deepfake_ai.models.classical.logistic_regression import build_logistic_regression


class ModelFactory:
    """Instantiates classical and deep learning models by name."""

    _MODELS = {
        'meso4': Meso4,
        'mesoinception4': MesoInception4,
        'xception': DeepfakeXception,
        'efficientnet': DeepfakeEfficientNet,
        'dual_stream_frequency': DualStreamFrequencyNet,
        'spatiotemporal_detector': SpatioTemporalDetector,
        'diffusion_detector': DiffusionResidualDetector,
    }

    _CLASSICAL = {
        'svm': build_svm,
        'random_forest': build_random_forest,
        'gradient_boosting': build_gradient_boosting,
        'logistic_regression': build_logistic_regression,
    }

    @classmethod
    def create(cls, name: str, **kwargs):
        name_lower = name.lower()
        if name_lower in cls._MODELS:
            return cls._MODELS[name_lower](**kwargs)
        elif name_lower in cls._CLASSICAL:
            return cls._CLASSICAL[name_lower](**kwargs)
        else:
            raise ModelError(f'Unknown model name: {name}. Available: {list(cls._MODELS.keys()) + list(cls._CLASSICAL.keys())}')
