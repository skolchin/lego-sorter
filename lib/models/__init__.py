# LEGO sorter project
# Model selection
# (c) lego-sorter team, 2022-2025

from lib.models.base import ModelBase, KerasModel
from lib.models.mobilenet import MobileNetModel
from lib.models.vgg19 import Vgg19Model
from lib.models.openblock import OpenBlockModel
from typing import Dict, Type

MODEL_BASES: Dict[str, Type[ModelBase]] = {
    cls()._model_class(): cls for cls in [MobileNetModel, Vgg19Model, OpenBlockModel]
}

