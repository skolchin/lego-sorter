# LEGO sorter project
# Custom MobileNet CNN model proxy class
# (c) lego-sorter team, 2022-2025

import numpy as np
from keras.applications.mobilenet_v3 import preprocess_input
from keras.applications import MobileNetV3Large
from typing import ClassVar, Any

from lib.models.custom import KerasModel, CustomModelBase

class MobileNetModel(CustomModelBase):
    def _model_instance(self, *args, **kwargs) -> KerasModel:
        """ Instantiate the model """
        return MobileNetV3Large(*args, **kwargs)
    
    def _model_class(self) -> str:
        """ Model class name """
        return 'mobile-net'

    def preprocess_input(self, input: Any, data_format: Any | None = None) -> Any:
        """ Convert images from a dataset format to model-specific format """
        return preprocess_input(input, data_format)

    def restore_processed_image(self, image: np.ndarray) -> np.ndarray:
        """ Convert images from a dataset format to something suitable for display """
        return image.astype('uint8')
    
