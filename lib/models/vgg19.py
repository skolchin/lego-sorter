# LEGO sorter project
# Custom VGG19 CNN model proxy class
# Based on standard  pretrained CNN architecture with additional layers to support transfer learning
# (c) lego-sorter team, 2022-2025

import numpy as np
from keras.applications.vgg19 import VGG19, preprocess_input
from typing import Any

from lib.models.custom import KerasModel, CustomModelBase

class Vgg19Model(CustomModelBase):

    def _model_instance(self, *args, **kwargs) -> KerasModel:
        """ Instantiate the model """
        return VGG19(*args, **kwargs)
    
    def _model_class(self) -> str:
        """ Model class name """
        return 'vgg19'

    def preprocess_input(self, input: Any, data_format: Any | None = None) -> Any:
        """ Convert images from a dataset format to model-specific format """
        return preprocess_input(input, data_format)

    def restore_processed_image(self, image: np.ndarray) -> np.ndarray:
        """ Convert images from a dataset format to something suitable for display """
        return image.astype('uint8')
