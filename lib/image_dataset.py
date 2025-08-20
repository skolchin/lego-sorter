# LEGO sorter project
# Image loading and processing functions
# (c) lego-sorter team, 2022-2025

import os
import numpy as np
import logging
import matplotlib
import keras as K
import tensorflow as tf
import tensorflow_models as tfm
import matplotlib.pyplot as plt
from absl import flags
from functools import partial
from itertools import zip_longest
from dataclasses import dataclass
from keras.utils import image_dataset_from_directory
from typing import Tuple, Iterable, Union, Callable, List, cast

from lib.globals import IMAGE_DIR, BATCH_SIZE
from lib.models.base import KerasModel, ModelProxy
from lib.img_utils import zoom_image, SizeT, ImageT

_logger = logging.getLogger(__name__)

matplotlib.use('Qt5Agg')

FLAGS = flags.FLAGS
flags.DEFINE_bool('gray', False, short_name='g',
    help='Convert images to grayscale')
flags.DEFINE_bool('edges', False, short_name='e',
    help='Convert images to wireframe (edges-only) images')
flags.DEFINE_bool('emboss', False, short_name='x',
    help='Mix wireframe with actual image, can be combined with --gray')
flags.DEFINE_boolean('zoom', False, short_name='z',
    help='Apply zoom augmentation (slows down the training by x5)')
flags.DEFINE_float('zoom_factor', 0.3,
    help='Maximum zoom level for image augmentation (for --zoom model only)')
flags.DEFINE_float('brightness_factor', 0.3,
    help='Maximum brightness level for image augmentation')
flags.DEFINE_float('rotation_factor', 0.45,
    help='Maximum rotation in image augmentation')
flags.DEFINE_float('crop_factor', 0.0, upper_bound=1.0,
    help='Crop image factor (from center)')
flags.DEFINE_float('edge_emboss', 0.5, lower_bound=0.0,
    help='Edge embossing factor (for --emboss model only)')
flags.DEFINE_float('shuffle_buffer_size', 0.25,
    help='Shuffle buffer, either ratio to number of files (<1) or actual size')

tf.get_logger().setLevel('ERROR')

@dataclass
class ImageDataset:
    """ Dataset with extra info resulting from `load_dataset()` call """
    tfds: tf.data.Dataset
    """ actual tensorflow dataset """

    num_files: int
    """ number of image files in a dataset """

    class_names: Iterable[str]
    """ class names """

def _sobel_edges(images):
    """ Internal - applies edges filter on gray image """
    shape = images.shape
    if len(shape) == 3: images = tf.expand_dims(images, 0)
    grad_components = tf.image.sobel_edges(images)
    grad_mag_components = grad_components**2
    grad_mag_square = tf.math.reduce_sum(grad_mag_components,axis=-1)
    images = tf.sqrt(grad_mag_square)
    if len(shape) == 3: images = images[0]
    return images

def _wireframe(images):
    """ Internal - generates wireframe image """
    shape = images.shape
    if len(shape) == 3: images = tf.expand_dims(images, 0)

    gray_images = tf.image.rgb_to_grayscale(images)
    sobel_images = _sobel_edges(gray_images)

    mask = tf.where(sobel_images < 127.5, 0.0, 255.0)   # VGG-16 operates on [0...255] range
    if FLAGS.edge_emboss > 0:
        kernel = tf.zeros((3, 3, mask.get_shape()[3]))
        dilation = (1.0, FLAGS.edge_emboss, FLAGS.edge_emboss, 1.0)
        mask = tf.nn.dilation2d(mask, kernel, (1,1,1,1), 'SAME', 'NHWC', dilation, 'dilation')

    if len(shape) == 3: mask = mask[0]
    return mask

def _preprocess(images, preprocess_fun: Callable, image_size: SizeT, labels=None, seed=None):
    """ Image preprocessing  """
    if FLAGS.crop_factor:
        images = tf.image.central_crop(images, FLAGS.crop_factor)
        images = tf.image.resize(images, image_size, preserve_aspect_ratio=False)

    images = preprocess_fun(images)

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
    num_samples = int(tf.shape(feature)[0]) #type:ignore
    degrees = tf.random.stateless_uniform(
        shape=(num_samples,), seed=seed, minval=FLAGS.rotation_factor*100.0, maxval=FLAGS.rotation_factor*100.0
    )
    degrees = degrees * 0.017453292519943295  # convert the angle in degree to radians
    rotated_images = tfm.vision.augment.rotate(feature, degrees)    #type:ignore
    return rotated_images, label, seed

@tf.function
def _zoom_images(feature, label, seed):
    def _func(images, seed):
        return K.layers.RandomZoom(
                    (-FLAGS.zoom_factor, FLAGS.zoom_factor),
                    (-FLAGS.zoom_factor, FLAGS.zoom_factor),
                    fill_mode='nearest', interpolation='nearest', seed=seed)(images)
    zoomed_images = tf.py_function(func=_func, inp=[feature, seed], Tout=tf.float32)
    return zoomed_images, label, seed

@tf.function
def _drop_seed(feature, label, seed):
    return feature, label

def load_dataset(model_proxy: ModelProxy) -> ImageDataset:
    """ Load images into a dataset """

    _flags = []
    if FLAGS.gray: _flags.append('gray')
    if FLAGS.edges: _flags.append('edges')
    if FLAGS.emboss: _flags.append('emboss')
    if FLAGS.zoom: _flags.append('zoom')
    _logger.debug(f'Model type flags: {", ".join(_flags)}')

    seed = np.random.randint(1, 1000000)
    ds: tf.data.Dataset = cast(tf.data.Dataset, image_dataset_from_directory(
        IMAGE_DIR,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=model_proxy.image_size(),
        shuffle=False,
        crop_to_aspect_ratio=True,
        seed=seed,
    ))
    assert hasattr(ds, 'file_paths')    # must present for directory-based datasets
    num_files: int = len(ds.file_paths) # type:ignore
    assert hasattr(ds, 'class_names')               # must present for directory-based datasets
    class_names: List[str] = ds.class_names.copy()  # type:ignore
    buf_size = int(FLAGS.shuffle_buffer_size * num_files) if FLAGS.shuffle_buffer_size < 1 \
        else int(FLAGS.shuffle_buffer_size)

    return ImageDataset(
        ds.shuffle(buffer_size=buf_size, reshuffle_each_iteration=False, seed=seed) \
            .map(partial(_preprocess,
                         preprocess_fun=model_proxy.preprocess_input,
                         image_size=model_proxy.image_size(),
                         seed=seed)) \
            .batch(BATCH_SIZE) \
            .prefetch(tf.data.AUTOTUNE),
        num_files,
        class_names,
    )

def split_dataset(ids: ImageDataset, split_by: float=0.8) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """ Split dataset into train/test TF subsets """
    train_size = int(ids.num_files * split_by / BATCH_SIZE)
    train_ds = ids.tfds.take(train_size)
    test_ds = ids.tfds.skip(train_size)
    return (train_ds, test_ds)

def augment_dataset(tfds: tf.data.Dataset) -> tf.data.Dataset:
    """ Applies multiple augmentatiions to the TF dataset """
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

def show_samples(tfds: tf.data.Dataset, model_proxy: ModelProxy, num_samples: int = 9):
    """ Show random image samples from TF dataset """
    plt.figure(figsize=(8, 8))
    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None
    class_names = model_proxy.get_class_labels()
    for images, labels in get_dataset_samples(tfds):    # pyright: ignore[reportGeneralTypeIssues]
        for i in range(num_samples):
            _ = plt.subplot(int(num_samples/3), int(num_samples/3), i + 1)
            plt.imshow(model_proxy.restore_processed_image(images[i].numpy()), cmap=cmap)
            label = np.argmax(labels[i])
            plt.title(class_names[label])
            plt.axis('off')
    plt.show()

def predict_image(
    model_proxy: ModelProxy,
    image: ImageT | tf.Tensor,
    return_image: bool = False) -> Tuple[str, float] | Tuple[str, float, tf.Tensor]:
    """ Run a pretrained model prediction on an image """

    resized_image = K.preprocessing.image.smart_resize(image, model_proxy.image_size())
    prepared_image, _ = _preprocess(
        tf.expand_dims(resized_image, 0),
        preprocess_fun=model_proxy.preprocess_input,
        image_size=model_proxy.image_size(),
    )

    prediction = model_proxy.keras_model.predict(prepared_image, verbose=0) # pyright: ignore[reportArgumentType]
    if isinstance(prediction, list):
        prediction = prediction[0]
    most_likely = np.argmax(prediction)
    label = model_proxy.get_class_labels()[most_likely]
    prob = prediction[0][most_likely]

    if not return_image:
        return label, prob

    return label, prob, prepared_image[0] # pyright: ignore[reportIndexIssue,reportOptionalSubscript]

def predict_image_probs(
    model_proxy: ModelProxy,
    image: ImageT) -> Iterable[Tuple[str, float]]:
    """ Run a pretrained model prediction on an image returning all detections and probabilities """

    resized_image = K.preprocessing.image.smart_resize(image, model_proxy.image_size())
    prepared_image, _ = _preprocess(
        tf.expand_dims(resized_image, 0),
        preprocess_fun=model_proxy.preprocess_input,
        image_size=model_proxy.image_size(),
    )

    prediction = model_proxy.keras_model.predict(prepared_image, verbose=0) # pyright: ignore[reportArgumentType]
    if isinstance(prediction, list):
        prediction = prediction[0]
    probs = []
    class_names = model_proxy.get_class_labels()
    for prob, pred in sorted(zip(prediction[0], range(len(prediction[0]))), key=lambda x: x[0], reverse=True):
        label = class_names[pred]
        probs.append((label, prob))
    return probs

def predict_dataset(
        model_proxy: ModelProxy,
        tfds: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor]:
    """ Get predictions for TF dataset """

    true_labels = [tf.argmax(labels) for _, labels in tfds.unbatch()] # pyright: ignore[reportGeneralTypeIssues]
    true_labels = tf.stack(true_labels, axis=0)

    predicted_labels = model_proxy.keras_model.predict(tfds)
    predicted_labels = tf.concat(predicted_labels, axis=0)
    predicted_labels = tf.argmax(predicted_labels, axis=1)

    return true_labels, predicted_labels

def predict_image_files_zoom(
    model_proxy: ModelProxy,
    file_names: Iterable[str],
    true_labels: Iterable[str] | None = None,
    zoom_levels: Iterable[float] | None = None,
    show_processed_image: bool = False):
    """ Run a pretrained model prediction on multiple image files with optional zooming and display the result """

    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None
    class_names = model_proxy.get_class_labels()
    for file_name, true_label in zip_longest(file_names, true_labels or []):
        image: tf.Tensor = cast(tf.Tensor, tf.image.decode_image(tf.io.read_file(file_name), channels=3))

        for zoom in zoom_levels or [1.0]:
            zoomed_image: tf.Tensor = cast(tf.Tensor, zoom_image(image, zoom, 255))
            zoomed_image = cast(tf.Tensor, tf.cast(zoomed_image, tf.uint8))

            predicted_label, predicted_prob, processed_image = predict_image( # pyright: ignore[reportAssignmentType]
                model_proxy, zoomed_image, return_image=True)

            plt.figure(f'{file_name} @ {zoom}')
            plt.title(f'{true_label or "?"} <- {predicted_label} ({predicted_prob:.2%}) @ {zoom}')
            if show_processed_image:
                img_to_show = model_proxy.restore_processed_image(processed_image.numpy()) # pyright: ignore[reportOptionalCall]
            else:
                img_to_show = zoomed_image.numpy()   # pyright: ignore[reportOptionalCall]

            plt.imshow(img_to_show, cmap=cmap)
            plt.axis('off')

    plt.show()

def predict_image_file(
    model_proxy: ModelProxy,
    file_name: str,
    true_label: str | None = None,
    show_processed_image: bool = False):
    """ Run a pretrained model prediction on an image file and display the result """

    predict_image_files_zoom(model_proxy, [file_name], [true_label] if true_label else None, show_processed_image=show_processed_image)

def predict_image_files(
    model_proxy: ModelProxy,
    file_names: Iterable[str],
    true_labels: Iterable[str] | None = None,
    show_processed_image: bool = False):
    """ Run a pretrained model prediction on multiple image files and display the result """

    predict_image_files_zoom(model_proxy, file_names, true_labels, show_processed_image=show_processed_image)

def show_prediction_samples(
    model_proxy: ModelProxy,
    tfds: tf.data.Dataset,
    num_samples: int = 9):
    """ Show prediction samples from test dataset """

    plt.figure(figsize=(8, 8))
    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None
    class_names = model_proxy.get_class_labels()
    for images, labels in get_dataset_samples(tfds):     # pyright: ignore[reportGeneralTypeIssues]
        for i in range(min(num_samples, labels.shape[0])):
            _ = plt.subplot(int(num_samples/3), int(num_samples/3), i + 1)
            label = class_names[np.argmax(labels[i])]
            image = images[i].numpy()
            prediction = model_proxy.keras_model.predict(np.array([image]))
            most_likely = np.argmax(prediction)
            predicted_label = class_names[most_likely]
            predicted_prob = prediction[0][most_likely]

            plt.title(f'{label} <- {predicted_label} ({predicted_prob:.2%})')
            plt.imshow(model_proxy.restore_processed_image(image), cmap=cmap)
            plt.axis('off')
    plt.show()

def filter_dataset_by_label(tfds: tf.data.Dataset, class_names: List[str], label: str) -> tf.data.Dataset:
    """ Return a dataset subset containing images of given label only """
    label_index = class_names.index(label)
    return tfds.unbatch().filter(lambda _, label: tf.equal(tf.math.argmax(label), label_index)).batch(BATCH_SIZE)

