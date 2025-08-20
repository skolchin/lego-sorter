# LEGO sorter project
# Custom trainable CNN model proxy class
# (c) lego-sorter team, 2022-2025

import os
import re
import logging
import numpy as np
import keras as K
import tensorflow as tf
from absl import flags
from pathlib import Path
from abc import abstractmethod
from contextlib import suppress
from keras import Model as KerasModel
from keras.callbacks import Callback as KerasCallback
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from typing import Mapping, Any, List, Tuple

from lib.globals import CHECKPOINT_DIR, IMAGE_DIR
from lib.models.base import ModelProxy

_logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_layers', default=2,
                     lower_bound=0, upper_bound=3,
                     help='Number of extra layers before classification head')
flags.DEFINE_integer('apply_gap', 0, lower_bound=0, upper_bound=1,
                      help='Apply GlobalAveragePooling2D to base model output')
flags.DEFINE_integer('units1', 512, help='Extra layer 1 size')
flags.DEFINE_integer('units2', 128, help='Extra layer 2 size')
flags.DEFINE_integer('units3', 64, help='Extra layer 3 size')
flags.DEFINE_float('dropout1', 0.3, help='Extra layer 1 dropout rate')
flags.DEFINE_float('dropout2', 0.3, help='Extra layer 2 dropout rate')
flags.DEFINE_float('dropout3', 0.3, help='Extra layer 3 dropout rate')
flags.DEFINE_float('regularize0', 0.01, help='Classification head regularization rate')
flags.DEFINE_float('regularize1', 0.01, help='Extra layer 1 regularizer rate')
flags.DEFINE_float('regularize2', 0.01, help='Extra layer 2 regularizer rate')
flags.DEFINE_float('regularize3', 0.01, help='Extra layer 3 regularizer rate')
flags.DEFINE_string('optimizer', 'Adam', help='Optimizer')
flags.DEFINE_float('learning_rate', 1e-3, help='Learning rate')
flags.DEFINE_float('momentum', 0, help='Momentum (for SGD optimizer only)')
flags.DEFINE_float('label_smoothing', 0.01, help='Label smoothing')

class CustomModelProxy(ModelProxy):

    @abstractmethod
    def _model_instance(self, *args, **kwargs) -> KerasModel:
        """ Instantiate the actual model """
        pass
    
    def image_size(self) -> Tuple[int,int]:
        return (224, 224)

    @property
    def supports_training(self) -> bool:
        return True

    def make_model_with_params(self, num_labels: int, params: Mapping[str, Any], fine_tuning: bool = False) -> KerasModel:
        """ Make and compile a Keras model with specified parameters.

        List of supported parameters see in FLAGS.
        """

        # Input shape
        if FLAGS.gray or FLAGS.edges:
            input_shape = list(self.image_size()) + [1]
            input_layer= K.layers.Input(input_shape)
            concat_layer= K.layers.Concatenate()([input_layer, input_layer, input_layer])
            last_layer = concat_layer
        else:
            input_shape = list(self.image_size()) + [3]
            input_layer= K.layers.Input(input_shape)
            last_layer = input_layer
        _logger.debug(f'Input shape set to {input_shape}')

        # Feature extractor
        base_model = self._model_instance(weights='imagenet', include_top=False, input_tensor=last_layer)
        _logger.debug(f'Using {base_model.__class__.__name__} model')

        if not fine_tuning:
            base_model.trainable = False
        else:
            _logger.debug('Model is been built in fine-tuning mode')
            for layer in base_model.layers[:16]:
                layer.trainable = False

        model = K.Sequential()
        model.add(input_layer)
        model.add(base_model)

        # GAP layer
        if params.get('apply_gap'):
            model.add(K.layers.GlobalAveragePooling2D())
            _logger.debug('GAP layer added')

        model.add(K.layers.Flatten())

        # Dense layers
        _logger.debug(f'Number of intermediate dense layers: {params["num_layers"]}')
        for n in range(int(params['num_layers'])):
            key = f'regularize{n+1}'
            reg = K.regularizers.l2(params[key]) \
                if key in params and params[key] > 0.0 else None

            key = f'units{n+1}'
            if not key in params:
                raise ValueError(f'Expected key {key} missing')
            if params[key] < num_labels:
                _logger.warning(f'Layer {n+1} size {params[key]} is less than number of labels {num_labels}')

            model.add(K.layers.Dense(int(params[key]), activation='relu', kernel_regularizer=reg))

            key = f'dropout{n+1}'
            if key in params and params[key] > 0.0:
                model.add(K.layers.Dropout(params[key]))

        # Final dense layer
        key = 'regularize0'
        l2_reg = K.regularizers.l2(params[key]) \
            if key in params and params[key] > 0.0 else None
        model.add(K.layers.Dense(num_labels, activation='softmax', kernel_regularizer=l2_reg))

        # Optimizer
        lr = params['learning_rate'] if not fine_tuning else 1e-5
        match params.get('optimizer', 'Adam').lower():
            case 'adam':
                opt = K.optimizers.Adam(learning_rate=lr)
            case 'sgd':
                opt = K.optimizers.SGD(learning_rate=lr, momentum=params['momentum'])
            case 'rmsprop':
                opt = K.optimizers.RMSprop(learning_rate=lr, momentum=params['momentum'])
            case _:
                raise ValueError(f'Unknown optimizer type: {params["optimizer"]}')
        _logger.debug(f'Using {params["optimizer"]} optimizer with LR={lr}')

        # Compile & build
        model.compile(
            optimizer=opt,  # type:ignore
            loss=K.losses.CategoricalCrossentropy(label_smoothing=params['label_smoothing']),
            metrics=[K.metrics.CategoricalAccuracy()])

        model.build([None] + list(input_shape))
        return model

    def get_class_labels(self) -> List[str]:
        """ Build up a list of supported class labels """
        return [f for f in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, f))]

    def make_model(self) -> KerasModel:
        """ Make and compile a Keras model with parameters defined in run-time FLAGS """

        num_labels = len(self.get_class_labels())
        return self.make_model_with_params(
            num_labels,
            params={
                'num_layers': FLAGS.num_layers,
                'apply_gap': FLAGS.apply_gap,
                'units1': FLAGS.units1,
                'units2': FLAGS.units2,
                'units3': FLAGS.units3,
                'dropout1': FLAGS.dropout1,
                'dropout2': FLAGS.dropout2,
                'dropout3': FLAGS.dropout3,
                'regularize0': FLAGS.regularize0,
                'regularize1': FLAGS.regularize1,
                'regularize2': FLAGS.regularize2,
                'regularize3': FLAGS.regularize3,
                'optimizer': FLAGS.optimizer,
                'momentum': FLAGS.momentum,
                'learning_rate': FLAGS.learning_rate,
                'label_smoothing': FLAGS.label_smoothing,
            },
            fine_tuning=self._fine_tuning
        )

    def load_from_checkpoint(self, checkpoint: str | Path | None = None) -> KerasModel:
        """ Load a model from latest checkpoint """

        # Make an empty model
        model = self.make_model()

        # Load from checkpoint
        if checkpoint:
            model.load_weights(checkpoint)
        else:
            cp_path = self.get_checkpoint_dir()
            cp_last = tf.train.latest_checkpoint(cp_path)
            if not cp_last:
                _logger.warning(f'No checkpoints found in {cp_path}')
            else:
                _logger.info(f'Loading model from {cp_last}')
                model.load_weights(cp_last)

        self._model = model
        return model

    def get_checkpoint_dir(self) -> str:
        match (FLAGS.gray, FLAGS.edges, FLAGS.emboss):
            case (False, False, False):
                subdir = 'color'
            case (True, False, False):
                subdir = 'gray'
            case (False, True, False):
                subdir = 'edges'
            case (False, False, True):
                subdir = 'emboss'
            case (True, False, True):
                subdir = 'emboss_gray'
            case _:
                raise ValueError('Invalid flags combination')

        if FLAGS.zoom: subdir += '_zoom'
        return os.path.join(CHECKPOINT_DIR, self._model_class(), subdir)

    def get_checkpoint_callback(self) -> KerasCallback:
        """ Build a callback to save weigths while fitting a model """
        return ModelCheckpoint(
            filepath=os.path.join(self.get_checkpoint_dir(), 'cp-{epoch:04d}.ckpt'),
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=0)

    def get_early_stopping_callback(self, patience: int = 5) -> KerasCallback:
        """ Build a callback to stop stale model fitting.

        This is basically a wrapper on Tensorflow's `EarlyStopping` callback for
        not importing tf package in the main script.
        """
        return EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=patience,
            restore_best_weights=True,
            verbose=0)

    def get_lr_reduce_callback(self, patience: int = 5) -> KerasCallback:
        """ Build a callback to reduce LR on plateau.

        This is basically a wrapper on Tensorflow's `ReduceLROnPlateau` callback for
        not importing tf package in the main script.
        """
        return ReduceLROnPlateau(
            monitor='val_categorical_accuracy', 
            factor=0.6, 
            patience=patience, 
            mode='max', 
            min_lr=1e-6,
            verbose=0)

    def cleanup_checkpoints(self, keep: int = 3):
        """ Remove abundant checkpoints after model fit """
        
        cp_path = self.get_checkpoint_dir()
        cp_last = tf.train.latest_checkpoint(cp_path)
        if not cp_last:
            _logger.warning(f'No checkpoints found in {cp_path}')
        else:
            cp_name = os.path.split(cp_last)[1]
            _logger.info(f'Latest checkpoint is {cp_name}')
            if (g := re.search('\\d+', cp_name)):
                cp_num = int(g.group(0))
            else:
                _logger.warning('Checkpoint file name does not contain ordinary number, cleanup is not possible')
                return

            files = [fn for fn in os.listdir(cp_path) if fn.startswith('cp-') and not cp_name in fn]
            for fn in files:
                if (g := re.search('\\d+', fn)):
                    ncp = int(g.group(0))
                    if ncp > cp_num or ncp < cp_num - keep:
                        with suppress(FileNotFoundError):
                            os.remove(os.path.join(cp_path, fn))

