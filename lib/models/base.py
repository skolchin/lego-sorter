# LEGO sorter project
# Base model proxy class
# (c) lego-sorter team, 2022-2025

import numpy as np
from abc import ABC, abstractmethod
from keras import Model as KerasModel
from typing import Any, Sequence, Tuple

class ModelBase(ABC):
    """ Base model class"""

    @abstractmethod
    def _model_class(self) -> str:
        """ Model class name """
        pass

    @abstractmethod
    def image_size(self) -> Tuple[int,int]:
        """ Image size """
        pass

    @abstractmethod
    def preprocess_input(self, input: Any, data_format: Any | None = None) -> Any:
        """ Convert images from a dataset format to model-specific format """
        pass

    @abstractmethod
    def restore_processed_image(self, image: np.ndarray) -> np.ndarray:
        """ Convert images from a dataset format to something suitable for display """
        pass

    @abstractmethod
    def get_class_labels(self) -> Sequence[str]:
        """ Build up a list of supported class labels """
        pass

    def make_model(self, fine_tuning: bool = False) -> KerasModel:
        """ Make and compile new Keras model """
        raise NotImplementedError('Not implemented')
    
    @abstractmethod
    def load_model(self, fine_tuning: bool = False) -> KerasModel:
        """ Load the model weights from latest checkpoint """
        pass

    @property
    def model_class(self) -> str:
        return self._model_class()

    @property
    def supports_training(self) -> bool:
        return False
