# LEGO sorter project
# CNN model definition
# Based on standard VGG-16 architecture with additional layers to support transfer learning
# (c) kol, 2022

import os
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input

BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
MODEL_DIR = os.path.join(os.path.split(__file__)[0], 'model')
CHECKPOINT_DIR = os.path.join(os.path.split(__file__)[0], 'checkpoints')

def make_model(num_labels: int, use_grayscale: bool = False) -> tf.keras.Model:
    """ Make and compile a DNN model """

    model: tf.keras.Model = tf.keras.models.Sequential()

    if not use_grayscale:
        input_shape = list(IMAGE_SIZE) + [3]
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        vgg16.trainable = False
        model.add(vgg16)
    else:
        input_shape = list(IMAGE_SIZE) + [1]
        input_layer = tf.keras.layers.Input(input_shape)
        concat_layer = tf.keras.layers.Concatenate()([input_layer, input_layer, input_layer])

        vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=concat_layer)
        vgg16.trainable = False

        model.add(input_layer)
        model.add(vgg16)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(num_labels, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy'])

    model.build([None] + list(input_shape))
    return model

def checkpoint_callback():
    """ Returns a checkpoint save callback to use with model.fit """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, 'cp-{epoch:04d}.ckpt'),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=0)

def load_model(model):
    """ Loads model from latest checkpoint """
    cp_path = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not cp_path:
        print('WARNING: no checkpoints found')
    else:
        print(f'Loading model from {cp_path}')
        model.load_weights(cp_path)
