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

import json
import numpy as np
import keras as K
from functools import cached_property
from typing import Any, Sequence, Tuple, Dict

from lib.globals import MODELS_DIR
from lib.models.base import ModelBase, KerasModel

class OpenBlockModel(ModelBase):
    """ Base model class"""

    @cached_property
    def _manifest(self) -> Dict[str, Any]:
        return json.loads((MODELS_DIR / self._model_class() / 'e2e_1680214477.json').read_text())

    def _model_class(self) -> str:
        """ Model class name """
        return 'openblock'

    def image_size(self) -> Tuple[int,int]:
        """ Image size """
        return self._manifest['image_height'], self._manifest['image_width']

    def preprocess_input(self, input: Any, data_format: Any | None = None) -> Any:
        """ Convert images from a dataset format to model-specific format """
        return input

    def restore_processed_image(self, image: np.ndarray) -> np.ndarray:
        """ Convert images from a dataset format to something suitable for display """
        return image

    def get_class_labels(self) -> Sequence[str]:
        """ Build up a list of supported class labels """
        return self._manifest['designs']

    def load_model(self, fine_tuning: bool = False) -> KerasModel:
        """ Load the model """
        m = K.models.load_model(MODELS_DIR / self._model_class() / 'e2e_1680214477.hdf5')
        m.compile()
        return m

    @property
    def model_class(self) -> str:
        return self._model_class()

    @property
    def supports_training(self) -> bool:
        return False
