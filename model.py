# LEGO sorter project
# CNN model definition and support functions
# Based on standard VGG-16 architecture with additional layers to support transfer learning
# (c) kol, 2022

import os
import global_flags
import tensorflow as tf
from absl import flags
from keras.applications.vgg16 import VGG16, preprocess_input

BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
MODEL_DIR = os.path.join(os.path.split(__file__)[0], 'model')
CHECKPOINT_DIR = os.path.join(os.path.split(__file__)[0], 'checkpoints')

FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-4, help='Learning rate')
flags.DEFINE_float('dropout_rate', 0.0, help='Dropout rate')
flags.DEFINE_float('regularizer_rate', 0.0, help='L2 regularizer rate')
flags.DEFINE_float('label_smoothing', 0.0, help='Label smoothing')

def _make_preproc_layers(last_layer):
    layers = [
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom((-0.2, 0.2)),
        # tf.keras.layers.RandomBrightness(0.2)
        # tf.keras.layers.RandomContrast(0.2)]
    ]
    for layer in layers:
        last_layer = layer(last_layer)
    return last_layer

def make_model(num_labels: int) -> tf.keras.Model:
    """ Make and compile a Keras model """

    model = tf.keras.models.Sequential()

    if not FLAGS.gray:
        input_shape = list(IMAGE_SIZE) + [3]
        input_layer = tf.keras.layers.Input(input_shape)
        last_preproc_layer = _make_preproc_layers(input_layer)
    else:
        input_shape = list(IMAGE_SIZE) + [1]
        input_layer = tf.keras.layers.Input(input_shape)
        concat_layer = tf.keras.layers.Concatenate()([input_layer, input_layer, input_layer])
        last_preproc_layer = _make_preproc_layers(concat_layer)

    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=last_preproc_layer)
    vgg16.trainable = False
    model.add(input_layer)
    model.add(vgg16)
    model.add(tf.keras.layers.Flatten())

    # Either dense or GAP layers should be used
    # model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    if FLAGS.dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(FLAGS.dropout_rate))

    if FLAGS.regularizer_rate > 0:
        model.add(tf.keras.layers.Dense(num_labels, activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(FLAGS.regularizer_rate)))
    else:
        model.add(tf.keras.layers.Dense(num_labels, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.experimental.SGD(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=FLAGS.learning_rate,
                decay_steps=1000,
                decay_rate=0.96,
            ),
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=FLAGS.label_smoothing),
        metrics=['accuracy'])

    model.build([None] + list(input_shape))
    return model

def _checkpoint_subdir():
    return 'gray' if FLAGS.gray else 'color'

def checkpoint_callback():
    """ Build a callback to save model weigth checkpoints while fitting a model """

    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, _checkpoint_subdir(), 'cp-{epoch:04d}.ckpt'),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=0)

def load_model(model: tf.keras.Model) -> tf.keras.Model:
    """ Load a model weights from latest checkpoint """

    cp_path = os.path.join(CHECKPOINT_DIR, _checkpoint_subdir())
    cp_last = tf.train.latest_checkpoint(cp_path)
    if not cp_last:
        print(f'WARNING: no checkpoints found in {cp_path}')
    else:
        print(f'Loading model from {cp_last}')
        model.load_weights(cp_last)

    return model

def cleanup_checkpoints():
    """ Remove abundant checkpoints after model fit run with checkpoint-saving callback """
    try:
        cp_path = os.path.join(CHECKPOINT_DIR, _checkpoint_subdir())
        cp_name = None
        with open(os.path.join(cp_path, 'checkpoint'), 'r') as fp:
            for line in fp.readlines():
                if 'model_checkpoint_path' in line:
                    cp_name = line.split(':')[1].strip().strip('\"')
        if cp_name:
            print(f'Latest checkpoint is {cp_name}')
            files = [fn for fn in os.listdir(cp_path) if fn.startswith('cp-') and not cp_name in fn]
            for fn in files:
                os.remove(os.path.join(cp_path, fn))
    except FileNotFoundError:
        pass
