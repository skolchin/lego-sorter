# LEGO sorter project
# Image loading and pre-processing functions (TF-dataset version)
# (c) kol, 2022

import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from absl import flags
from keras.utils import image_dataset_from_directory
from collections import namedtuple
from functools import partial
from typing import Tuple, Iterable

from lib.globals import IMAGE_DIR, BATCH_SIZE, IMAGE_SIZE
from lib.model import preprocess_input

FLAGS = flags.FLAGS
flags.DEFINE_float('zoom_factor', 0.3, help='Maximum zoom level for image augmentation')
flags.DEFINE_float('brightness_factor', 0.3, help='Maximum brightness level for image augmentation')
flags.DEFINE_float('rotation_factor', 0.45, help='Maximum rotation in image augmentation')
flags.DEFINE_integer('edge_emboss', 1, lower_bound=0, help='Edge embossing factor')

tf.get_logger().setLevel('ERROR')

ImageDataset = namedtuple('ImageDataset', ['tfds', 'num_files', 'class_names'])
""" Dataset with extra info """

def _sobel_edges(images):
    shape = images.shape
    if len(shape) == 3: images = tf.expand_dims(images, 0)
    grad_components = tf.image.sobel_edges(images)
    grad_mag_components = grad_components**2
    grad_mag_square = tf.math.reduce_sum(grad_mag_components,axis=-1)
    images = tf.sqrt(grad_mag_square)
    if len(shape) == 3: images = images[0]
    return images

def _wireframe(images):
    shape = images.shape
    if len(shape) == 3: images = tf.expand_dims(images, 0)

    gray_images = tf.image.rgb_to_grayscale(images)
    sobel_images = _sobel_edges(gray_images)

    mask = tf.where(sobel_images < 127.5, 0.0, 255.0)   # VGG-16 operates on [0...255] range
    if FLAGS.edge_emboss > 0:
        kernel = tf.zeros((3, 3, mask.get_shape()[3]))
        dilation = (1, FLAGS.edge_emboss, FLAGS.edge_emboss, 1)
        mask = tf.nn.dilation2d(mask, kernel, (1,1,1,1), 'SAME', 'NHWC', dilation, 'dilation')

    if len(shape) == 3: mask = mask[0]
    return mask

def _preprocess(images, labels=None, seed=None):
    """ Image preprocessing function """
    images = preprocess_input(images)

    if FLAGS.emboss:
        mask = _wireframe(images)
        images = tf.where(tf.equal(mask, 0.0), images, tf.constant((0.0, 0.0, 255.0), images.dtype))
        if FLAGS.gray:
            images = tf.image.rgb_to_grayscale(images)
    elif FLAGS.gray:
        images = tf.image.rgb_to_grayscale(images)
    elif FLAGS.edges:
        images = _wireframe(images)

    return images, labels

# https://towardsdatascience.com/image-augmentations-in-tensorflow-62967f59239d
@tf.function
def _augment_images(image_and_label, seed):
    image, label = image_and_label
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    image = tf.image.stateless_random_brightness(image, max_delta=FLAGS.brightness_factor, seed=new_seed)
    image = tf.image.stateless_random_flip_left_right(image, seed=new_seed)
    image = tf.image.stateless_random_flip_up_down(image, seed=new_seed)
    return image, label, seed

@tf.function
def _rotate_images(feature, label, seed):
    num_samples = int(tf.shape(feature)[0])
    degrees = tf.random.stateless_uniform(
        shape=(num_samples,), seed=seed, minval=FLAGS.rotation_factor*100, maxval=FLAGS.rotation_factor*100
    )
    degrees = degrees * 0.017453292519943295  # convert the angle in degree to radians
    rotated_images = tfa.image.rotate(feature, degrees, fill_mode='reflect')
    return rotated_images, label, seed

@tf.function
def _zoom_images(feature, label, seed):
    def _func(images, seed):
        return tf.keras.layers.RandomZoom(
                    (-FLAGS.zoom_factor, FLAGS.zoom_factor),
                    (-FLAGS.zoom_factor, FLAGS.zoom_factor),
                    fill_mode='nearest', interpolation='nearest', seed=seed)(images)
    zoomed_images = tf.py_function(func=_func, inp=[feature, seed], Tout=tf.float32)
    return zoomed_images, label, seed

@tf.function
def _drop_seed(feature, label, seed):
    return feature, label

def load_dataset() -> ImageDataset:
    """ Load images as dataset """
    seed = np.random.randint(1e6)
    ds = image_dataset_from_directory(
        IMAGE_DIR,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=None,
        image_size=IMAGE_SIZE,
        shuffle=False,
        crop_to_aspect_ratio=True,
        seed=seed,
    )
    num_files = len(ds.file_paths)
    return ImageDataset(
        ds.shuffle(buffer_size=int(num_files/4), reshuffle_each_iteration=False, seed=seed) \
            .map(partial(_preprocess, seed=seed)) \
            .batch(BATCH_SIZE) \
            .prefetch(tf.data.AUTOTUNE), 
        num_files,
        ds.class_names.copy()
    )

def split_dataset(ids: ImageDataset, split_by: float=0.8) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """ Split dataset into train/test TF subsets """
    train_size = int(ids.num_files * split_by / BATCH_SIZE)
    train_ds = ids.tfds.take(train_size)
    test_ds = ids.tfds.skip(train_size)
    return (train_ds, test_ds)

def augment_dataset(tfds: tf.data.Dataset) -> tf.data.Dataset:
    counter = tf.data.experimental.Counter()
    tfds_with_seed = tf.data.Dataset.zip((tfds, (counter, counter)))
    augmented_dataset = tfds_with_seed.map(_augment_images)
    augmented_dataset = augmented_dataset.map(_rotate_images)
    if FLAGS.zoom:
        augmented_dataset = augmented_dataset.map(_zoom_images)
    augmented_dataset = augmented_dataset.map(_drop_seed)
    return augmented_dataset

def get_dataset_samples(tfds: tf.data.Dataset, num_batches: int = 1) -> tf.data.Dataset:
    """ Generate given number of samples from dataset """
    shape = tfds.cardinality().numpy()
    if shape <= num_batches:
        return tfds

    sample_idx = np.random.randint(shape-num_batches)
    return tfds.take(sample_idx).skip(sample_idx-num_batches)

def show_samples(tfds: tf.data.Dataset, class_names: Iterable[str], num_samples: int = 9):
    """ Show random image samples from TF dataset """
    plt.figure(figsize=(8, 8))
    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None
    for images, labels in get_dataset_samples(tfds):
        for i in range(num_samples):
            _ = plt.subplot(int(num_samples/3), int(num_samples/3), i + 1)
            plt.imshow(images[i].numpy().astype('uint8'), cmap=cmap)
            label = np.argmax(labels[i])
            plt.title(class_names[label])
            plt.axis('off')
    plt.show()

def predict_image(
    model: tf.keras.Model, 
    image: np.ndarray, 
    class_names: Iterable[str]) -> Tuple[str, float]:
    """ Run a pretrained model prediction on an image """

    resized_image = tf.keras.preprocessing.image.smart_resize(image, IMAGE_SIZE)
    prepared_image, _ = _preprocess(tf.expand_dims(resized_image, 0))

    prediction = model.predict(prepared_image, verbose=0)
    most_likely = np.argmax(prediction)
    label = class_names[most_likely]
    prob = prediction[0][most_likely]

    return label, prob

def predict_image_probs(
    model: tf.keras.Model, 
    image: np.ndarray, 
    class_names: Iterable[str]) -> Iterable[Tuple[str, float]]:
    """ Run a pretrained model prediction on an image """

    resized_image = tf.keras.preprocessing.image.smart_resize(image, IMAGE_SIZE)
    prepared_image, _ = _preprocess(tf.expand_dims(resized_image, 0))

    prediction = model.predict(prepared_image, verbose=0)
    probs = []
    for prob, pred in sorted(zip(prediction[0], range(len(prediction[0]))), key=lambda x: x[0], reverse=True):
        label = class_names[pred]
        probs.append((label, prob))
    return probs

def predict_image_file(
    model: tf.keras.Model, 
    file_name: str, 
    class_names: Iterable[str], 
    true_label: str = None):
    """ Run a pretrained model prediction on an image file and display the result """

    predict_image_files(model, [file_name], class_names, [true_label])

def predict_image_files(
    model: tf.keras.Model, 
    file_names: Iterable[str], 
    class_names: Iterable[str], 
    true_labels: Iterable[str] = None):
    """ Run a pretrained model prediction on multiple image files and display the result """

    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None
    for file_name, true_label in zip(file_names, true_labels):
        image = tf.image.decode_image(tf.io.read_file(file_name))
        predicted_label, predicted_prob = predict_image(model, image, class_names)

        plt.figure(file_name)
        plt.title(f'{true_label or "?"} <- {predicted_label} ({predicted_prob:.2%})')
        plt.imshow(image.astype('uint8'), cmap=cmap)
        plt.axis('off')

    plt.show()

def predict_dataset(model: tf.keras.Model, tfds: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor]:
    """ Get predictions for TF dataset """

    true_labels = [tf.argmax(labels) for _, labels in tfds.unbatch()]
    true_labels = tf.stack(true_labels, axis=0)

    predicted_labels = model.predict(tfds)
    predicted_labels = tf.concat(predicted_labels, axis=0)
    predicted_labels = tf.argmax(predicted_labels, axis=1)

    return true_labels, predicted_labels

def show_prediction_samples(
    model: tf.keras.Model, 
    tfds: tf.data.Dataset,
    class_names: Iterable[str],
    num_samples: int = 9):
    """ Show prediction samples from test dataset """
    
    plt.figure(figsize=(8, 8))
    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None
    for images, labels in get_dataset_samples(tfds):
        for i in range(num_samples):
            _ = plt.subplot(int(num_samples/3), int(num_samples/3), i + 1)
            label = class_names[np.argmax(labels[i])]
            image = images[i].numpy()
            prediction = model.predict(np.array([image]))
            most_likely = np.argmax(prediction)
            predicted_label = class_names[most_likely]
            predicted_prob = prediction[0][most_likely]

            plt.title(f'{label} <- {predicted_label} ({predicted_prob:.2%})')
            plt.imshow(image.astype('uint8'), cmap=cmap)
            plt.axis('off')
    plt.show()

def filter_dataset_by_label(tfds: tf.data.Dataset, class_names: Iterable[str], label: str) -> tf.data.Dataset:
    """ Return a dataset subset containing images of given label only """
    label_index = class_names.index(label)
    return tfds.unbatch().filter(lambda _, label: tf.equal(tf.math.argmax(label), label_index)).batch(BATCH_SIZE)

def fast_get_class_names():
    """ Faster way of getting image class names """
    return [f for f in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, f))]
