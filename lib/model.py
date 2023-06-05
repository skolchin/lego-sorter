# LEGO sorter project
# CNN model definition and support functions
# Based on standard  pretrained CNN architecture with additional layers to support transfer learning
# (c) lego-sorter team, 2022-2023

import os
import tensorflow as tf
import logging
from absl import flags
from typing import Mapping
from lib.globals import IMAGE_SIZE, CHECKPOINT_DIR

# from keras.applications.vgg19 import VGG19, preprocess_input
# MODEL_CLASS = VGG19
# restore_processed_image = lambda x: x.astype('uint8')
""" This model-specific lambda func is used to convert images from a dataset format to something suitable for display """

# from keras.applications.inception_v3 import InceptionV3, preprocess_input
# MODEL_CLASS = InceptionV3
# restore_processed_image = lambda x: (127.5 * (x + 1.0)).astype('uint8')

from keras.applications.mobilenet_v3 import MobileNetV3Large, preprocess_input
MODEL_CLASS = MobileNetV3Large
restore_processed_image = lambda x: x.astype('uint8')

_logger = logging.getLogger('lego-sorter')

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

def make_model_with_params(num_labels: int, params: Mapping[str, float], fine_tuning: bool = False) -> tf.keras.Model:
    """ Make and compile a Keras model with specified parameters.

    List of supported parameters see in FLAGS.
    """

    # Input shape
    if FLAGS.gray or FLAGS.edges:
        input_shape = list(IMAGE_SIZE) + [1]
        input_layer = tf.keras.layers.Input(input_shape)
        concat_layer = tf.keras.layers.Concatenate()([input_layer, input_layer, input_layer])
        last_layer = concat_layer
    else:
        input_shape = list(IMAGE_SIZE) + [3]
        input_layer = tf.keras.layers.Input(input_shape)
        last_layer = input_layer
    _logger.debug(f'Input shape set to {input_shape}')

    # Feature extractor
    base_model = MODEL_CLASS(weights='imagenet', include_top=False, input_tensor=last_layer)
    _logger.debug(f'Using {MODEL_CLASS.__name__} model')

    if not fine_tuning:
        base_model.trainable = False
    else:
        _logger.debug('Model is been built in fine-tuning mode')
        for layer in base_model.layers[:16]:
            layer.trainable = False

    model = tf.keras.models.Sequential()
    model.add(input_layer)
    model.add(base_model)

    # GAP layer
    if params.get('apply_gap'):
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        _logger.debug('GAP layer added')

    model.add(tf.keras.layers.Flatten())

    # Dense layers
    _logger.debug(f'Number of intermediate dense layers: {params["num_layers"]}')
    for n in range(params['num_layers']):
        key = f'regularize{n+1}'
        reg = tf.keras.regularizers.l2(params[key]) \
            if key in params and params[key] > 0.0 else None

        key = f'units{n+1}'
        if not key in params:
            raise ValueError(f'Expected key {key} missing')
        if params[key] < num_labels:
            _logger.warning(f'Layer {n+1} size {params[key]} is less than number of labels {num_labels}')

        model.add(tf.keras.layers.Dense(int(params[key]), activation='relu', kernel_regularizer=reg))

        key = f'dropout{n+1}'
        if key in params and params[key] > 0.0:
            model.add(tf.keras.layers.Dropout(params[key]))

    # Final dense layer
    key = 'regularize0'
    l2_reg = tf.keras.regularizers.l2(params[key]) \
        if key in params and params[key] > 0.0 else None
    model.add(tf.keras.layers.Dense(num_labels, activation='softmax', kernel_regularizer=l2_reg))

    # Optimizer
    lr = params['learning_rate'] if not fine_tuning else 1e-5
    match params.get('optimizer', 'Adam').lower():
        case 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        case 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=params['momentum'])
        case 'rmsprop':
            opt = tf.keras.optimizers.experimental.RMSprop(learning_rate=lr, momentum=params['momentum'])
        case _:
            raise ValueError(f'Unknown optimizer type: {params["optimizer"]}')
    _logger.debug(f'Using {params["optimizer"]} optimizer with LR={lr}')

    # Compile & build
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=params['label_smoothing']),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])

    model.build([None] + list(input_shape))
    return model

def make_model(num_labels: int, fine_tuning: bool = False) -> tf.keras.Model:
    """ Make and compile a Keras model with parameters defined in run-time FLAGS """

    return make_model_with_params(
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
        fine_tuning=fine_tuning
    )

def _get_checkpoint_dir() -> str:
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
    model_class = MODEL_CLASS.__name__.lower()
    return os.path.join(CHECKPOINT_DIR, model_class, subdir)

def get_checkpoint_callback() -> tf.keras.callbacks.Callback:
    """ Build a callback to save weigths while fitting a model """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(_get_checkpoint_dir(), 'cp-{epoch:04d}.ckpt'),
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=0)

def get_early_stopping_callback(patience: int = 5) -> tf.keras.callbacks.Callback:
    """ Build a callback to stop stale model fitting.

    This is basically a wrapper on Tensorflow's `EarlyStopping` callback for
    not importing tf package in the main script.
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=patience,
        restore_best_weights=True,
        verbose=0)

def get_lr_reduce_callback(patience: int = 5) -> tf.keras.callbacks.Callback:
    """ Build a callback to reduce LR on plateau.

    This is basically a wrapper on Tensorflow's `ReduceLROnPlateau` callback for
    not importing tf package in the main script.
    """
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_categorical_accuracy', 
        factor=0.6, 
        patience=patience, 
        mode='max', 
        min_lr=1e-6,
        verbose=0)

def load_model(model: tf.keras.Model) -> tf.keras.Model:
    """ Load a model weights from latest checkpoint """

    cp_path = _get_checkpoint_dir()
    cp_last = tf.train.latest_checkpoint(cp_path)
    if not cp_last:
        _logger.warning(f'No checkpoints found in {cp_path}')
    else:
        _logger.info(f'Loading model from {cp_last}')
        model.load_weights(cp_last)

    return model

def cleanup_checkpoints(keep: int = 3):
    """ Remove abundant checkpoints after model fit """
    import re
    from contextlib import suppress
    
    cp_path = _get_checkpoint_dir()
    cp_last = tf.train.latest_checkpoint(cp_path)
    if not cp_last:
        _logger.warning(f'No checkpoints found in {cp_path}')
    else:
        cp_name = os.path.split(cp_last)[1]
        _logger.info(f'Latest checkpoint is {cp_name}')
        cp_num = int(re.search('\\d+', cp_name).group(0))

        files = [fn for fn in os.listdir(cp_path) if fn.startswith('cp-') and not cp_name in fn]
        for fn in files:
            ncp = int(re.search('\\d+', fn).group(0))
            if ncp > cp_num or ncp < cp_num - keep:
                with suppress(FileNotFoundError):
                    os.remove(os.path.join(cp_path, fn))
