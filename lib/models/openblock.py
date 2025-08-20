# LEGO sorter project
# OpenBlock model proxy class
# (c) lego-sorter team, 2022-2025
#
# For Keras > 2.11 use `export TF_USE_LEGACY_KERAS=1` + patch a file:
#
#   .venv/lib/python3.10/site-packages/keras/src/ops/operation.py:254
#           try:
#               return cls(**config)
#       ->
#           try:
#               if cls.__name__ in ["Conv2DTranspose", "DepthwiseConv2D"] and "groups" in config: 
#                   config.pop("groups")            
#               return cls(**config)

import cv2
import json
import numpy as np
import keras as K
import tensorflow as tf
from pathlib import Path
from functools import cached_property
from tensorflow.keras.models import load_model  # type:ignore
from typing import Any, List, Tuple, Dict

from lib.globals import MODELS_DIR
from lib.models.base import ModelProxy, KerasModel

class OpenBlockModel(ModelProxy):
    """ Base model class"""

    @cached_property
    def _manifest(self) -> Dict[str, Any]:
        return json.loads((MODELS_DIR / self._model_class() / 'e2e_1680214477.json').read_text())

    def _model_class(self) -> str:
        """ Model class name """
        return 'openblock'

    def image_size(self) -> Tuple[int,int]:
        """ Image size """
        return self._manifest['image_width'], self._manifest['image_height']

    @property
    def model_class(self) -> str:
        return self._model_class()

    @property
    def supports_training(self) -> bool:
        return False
    
    def preprocess_input(self, input: Any, data_format: Any | None = None) -> Any:
        """ Convert images from a dataset format to model-specific format """
        sz = self.image_size()
        if isinstance(input, np.ndarray):
            input = cv2.resize(input, sz, interpolation=cv2.INTER_CUBIC)
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            input = tf.convert_to_tensor(input, dtype=tf.float32)
            img_array = K.utils.img_to_array(input) / 255.0
            img_array = tf.expand_dims(img_array, 0)
            return img_array
        
        if isinstance(input, tf.Tensor):
            img = tf.cast(input, tf.float32)
            img = tf.divide(img, 255.0)
            img = tf.image.resize(img, (sz[1], sz[0]), 'bicubic', preserve_aspect_ratio=False)
            return img
    
        raise ValueError(f'Unknown input type {type(input)}')

    def restore_processed_image(self, image: np.ndarray) -> np.ndarray:
        """ Convert images from a dataset format to something suitable for display """
        return image

    def get_class_labels(self) -> List[str]:
        """ Build up a list of supported class labels """
        return self._manifest['designs']

    def load_from_checkpoint(self, checkpoint: str | Path | None = None) -> KerasModel:
        """ Load the model """
        model = load_model(MODELS_DIR / self._model_class() / 'e2e_1680214477.hdf5')
        if isinstance(model, KerasModel):
            model.compile()
            
        self._model = model
        return model
