# LEGO sorter project
# Base model proxy class
# (c) lego-sorter team, 2022-2025

import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from keras import Model as KerasModel
from typing import Any, List, Tuple

class ModelProxy(ABC):
    """ Base model proxy class"""

    _model: KerasModel | None
    _fine_tuning: bool

    def __init__(self, fine_tuning: bool = False):
        self._model = None
        self._fine_tuning = fine_tuning

    @property
    def model_class(self) -> str:
        return self._model_class()

    @property
    def supports_training(self) -> bool:
        return False

    @property
    def keras_model(self) -> KerasModel:
        if self._model is not None:
            return self._model
        
        m = self.make_model()
        self._model = m
        return m

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
    def get_class_labels(self) -> List[str]:
        """ Build up a list of supported class labels """
        pass

    def make_model(self) -> KerasModel:
        """ Make and compile new Keras model """
        raise NotImplementedError('Not implemented')
    
    @abstractmethod
    def load_from_checkpoint(self, checkpoint: str | Path | None = None) -> KerasModel:
        """ Load the model weights from specified or latest checkpoint """
        pass
