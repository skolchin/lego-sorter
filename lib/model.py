# LEGO sorter project
# CNN model definition and support functions
# Based on standard VGG-16 architecture with additional layers to support transfer learning
# (c) kol, 2022

import os
import tensorflow as tf
import logging
from absl import flags
from keras.applications.vgg16 import VGG16, preprocess_input      # pylint: disable=unused-import

from lib.globals import IMAGE_SIZE, CHECKPOINT_DIR

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-4, help='Learning rate')
flags.DEFINE_float('dropout_rate', 0.3, help='Dropout rate')
flags.DEFINE_float('regularizer_rate', 0.01, help='L2 regularizer rate')
flags.DEFINE_float('label_smoothing', 0.01, help='Label smoothing')

def make_model(num_labels: int) -> tf.keras.Model:
    """ Make and compile a Keras model """

    if not FLAGS.gray:
        input_shape = list(IMAGE_SIZE) + [3]
        input_layer = tf.keras.layers.Input(input_shape)
        last_layer = input_layer
    else:
        input_shape = list(IMAGE_SIZE) + [1]
        input_layer = tf.keras.layers.Input(input_shape)
        concat_layer = tf.keras.layers.Concatenate()([input_layer, input_layer, input_layer])
        last_layer = concat_layer

    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=last_layer)
    vgg16.trainable = False

    model = tf.keras.models.Sequential()
    model.add(input_layer)
    model.add(vgg16)
    model.add(tf.keras.layers.Flatten())

    # Either dense or GAP layers should be used
    # model.add(tf.keras.layers.GlobalAveragePooling2D())

    l2_reg = tf.keras.regularizers.l2(FLAGS.regularizer_rate) if FLAGS.regularizer_rate > 0 else None
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2_reg))
    if FLAGS.dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(FLAGS.dropout_rate))

    l2_reg = tf.keras.regularizers.l2(FLAGS.regularizer_rate) if FLAGS.regularizer_rate > 0 else None
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2_reg))
    if FLAGS.dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(FLAGS.dropout_rate))

    l2_reg = tf.keras.regularizers.l2(FLAGS.regularizer_rate) if FLAGS.regularizer_rate > 0 else None
    model.add(tf.keras.layers.Dense(num_labels, activation='softmax', kernel_regularizer=l2_reg))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=FLAGS.learning_rate,
                decay_steps=1000,
                decay_rate=0.96,
            ),
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=FLAGS.label_smoothing),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])

    model.build([None] + list(input_shape))
    return model

def _checkpoint_subdir():
    match (FLAGS.gray, FLAGS.edges):
        case (True, True):
            return 'gray_edges'
        case (True, False):
            return 'gray'
        case (False, True):
            return 'color_edges'
        case (False, False):
            return 'color'

def checkpoint_callback():
    """ Build a callback to save model weigth checkpoints while fitting a model """

    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, _checkpoint_subdir(), 'cp-{epoch:04d}.ckpt'),
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=0)

def load_model(model: tf.keras.Model) -> tf.keras.Model:
    """ Load a model weights from latest checkpoint """

    cp_path = os.path.join(CHECKPOINT_DIR, _checkpoint_subdir())
    cp_last = tf.train.latest_checkpoint(cp_path)
    if not cp_last:
        logger.warning(f'No checkpoints found in {cp_path}')
    else:
        logger.info(f'Loading model from {cp_last}')
        model.load_weights(cp_last)

    return model

def cleanup_checkpoints():
    """ Remove abundant checkpoints after model fit run with checkpoint-saving callback """
    try:
        cp_path = os.path.join(CHECKPOINT_DIR, _checkpoint_subdir())
        cp_last = tf.train.latest_checkpoint(cp_path)
        if not cp_last:
            logger.warning(f'No checkpoints found in {cp_path}')
        else:
            cp_name = os.path.split(cp_last)[1]
            logger.info(f'Latest checkpoint is {cp_name}')
            files = [fn for fn in os.listdir(cp_path) if fn.startswith('cp-') and not cp_name in fn]
            for fn in files:
                os.remove(os.path.join(cp_path, fn))
    except FileNotFoundError:
        pass
