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
from typing import Tuple, Iterable, Union, Callable

from lib.globals import IMAGE_DIR, BATCH_SIZE
from lib.models.base import KerasModel, ModelBase

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

def _preprocess(images, preprocess_fun: Callable, image_size: Tuple[int,int], labels=None, seed=None):
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
    num_samples = int(tf.shape(feature)[0])
    degrees = tf.random.stateless_uniform(
        shape=(num_samples,), seed=seed, minval=FLAGS.rotation_factor*100.0, maxval=FLAGS.rotation_factor*100.0
    )
    degrees = degrees * 0.017453292519943295  # convert the angle in degree to radians
    rotated_images = tfm.vision.augment.rotate(feature, degrees)
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

def load_dataset(model_base: ModelBase) -> ImageDataset:
    """ Load images into a dataset """

    _flags = []
    if FLAGS.gray: _flags.append('gray')
    if FLAGS.edges: _flags.append('edges')
    if FLAGS.emboss: _flags.append('emboss')
    if FLAGS.zoom: _flags.append('zoom')
    _logger.debug(f'Model type flags: {", ".join(_flags)}')

    seed = np.random.randint(1, 1000000)
    ds = image_dataset_from_directory(
        IMAGE_DIR,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=model_base.image_size(),
        shuffle=False,
        crop_to_aspect_ratio=True,
        seed=seed,
    )
    num_files = len(ds.file_paths)  # type:ignore
    buf_size = int(FLAGS.shuffle_buffer_size * num_files) if FLAGS.shuffle_buffer_size < 1 \
        else int(FLAGS.shuffle_buffer_size)
    
    return ImageDataset(
        ds.shuffle(buffer_size=buf_size, reshuffle_each_iteration=False, seed=seed) \
            .map(partial(_preprocess, 
                         preprocess_fun=model_base.preprocess_input, 
                         image_size=model_base.image_size(),
                         seed=seed)) \
            .batch(BATCH_SIZE) \
            .prefetch(tf.data.AUTOTUNE),
        num_files,
        ds.class_names.copy() # type:ignore
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

def show_samples(tfds: tf.data.Dataset, model_base: ModelBase, num_samples: int = 9):
    """ Show random image samples from TF dataset """
    plt.figure(figsize=(8, 8))
    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None
    class_names = model_base.get_class_labels()
    for images, labels in get_dataset_samples(tfds):
        for i in range(num_samples):
            _ = plt.subplot(int(num_samples/3), int(num_samples/3), i + 1)
            plt.imshow(model_base.restore_processed_image(images[i].numpy()), cmap=cmap)
            label = np.argmax(labels[i])
            plt.title(class_names[label])
            plt.axis('off')
    plt.show()

def predict_image(
    model_base: ModelBase, 
    model: KerasModel,
    image: np.ndarray,
    return_image: bool = False) -> Tuple[str, float, np.ndarray]:
    """ Run a pretrained model prediction on an image """

    resized_image = K.preprocessing.image.smart_resize(image, model_base.image_size())
    prepared_image, _ = _preprocess(
        tf.expand_dims(resized_image, 0), 
        preprocess_fun=model_base.preprocess_input,
        image_size=model_base.image_size(),
    )

    prediction = model.predict(prepared_image, verbose=0)
    if isinstance(prediction, list):
        prediction = prediction[0]
    most_likely = np.argmax(prediction)
    label = model_base.get_class_labels()[most_likely]
    prob = prediction[0][most_likely]

    if not return_image:
        return label, prob

    return label, prob, prepared_image[0]

def predict_image_probs(
    model_base: ModelBase, 
    model: KerasModel,
    image: np.ndarray) -> Iterable[Tuple[str, float]]:
    """ Run a pretrained model prediction on an image returning all detections and probabilities """

    resized_image = K.preprocessing.image.smart_resize(image, model_base.image_size())
    prepared_image, _ = _preprocess(
        tf.expand_dims(resized_image, 0), 
        preprocess_fun=model_base.preprocess_input,
        image_size=model_base.image_size(),
    )

    prediction = model.predict(prepared_image, verbose=0)
    if isinstance(prediction, list):
        prediction = prediction[0]
    probs = []
    class_names = model_base.get_class_labels()
    for prob, pred in sorted(zip(prediction[0], range(len(prediction[0]))), key=lambda x: x[0], reverse=True):
        label = class_names[pred]
        probs.append((label, prob))
    return probs

def predict_dataset(
        model_base: ModelBase, 
        model: KerasModel, 
        tfds: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor]:
    """ Get predictions for TF dataset """

    true_labels = [tf.argmax(labels) for _, labels in tfds.unbatch()]
    true_labels = tf.stack(true_labels, axis=0)

    predicted_labels = model.predict(tfds)
    predicted_labels = tf.concat(predicted_labels, axis=0)
    predicted_labels = tf.argmax(predicted_labels, axis=1)

    return true_labels, predicted_labels

def _pad_to_bounding_box_color(image, offset_height, offset_width,
                                 target_height, target_width, pad_color):
  """ Internal - TF `pad_to_bounding_box()` function adjusted to accept padding color """
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import array_ops
  from tensorflow.python.ops.image_ops_impl import _ImageDimensions, _is_tensor

  with ops.name_scope(None, 'pad_to_bounding_box_white', [image]):
    image = ops.convert_to_tensor(image, name='image')

    is_batch = True
    image_shape = image.get_shape()
    if image_shape.ndims == 3:
      is_batch = False
      image = array_ops.expand_dims(image, 0)
    elif image_shape.ndims is None:
      is_batch = False
      image = array_ops.expand_dims(image, 0)
      image.set_shape([None] * 4)
    elif image_shape.ndims != 4:
      raise ValueError(
          '\'image\' (shape %s) must have either 3 or 4 dimensions.' %
          image_shape)

    batch, height, width, depth = _ImageDimensions(image, rank=4)
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height

    # Do not pad on the depth dimensions.
    paddings = array_ops.reshape(
        array_ops.stack([   #type:ignore
            0, 0, offset_height, after_padding_height, offset_width,
            after_padding_width, 0, 0
        ]), [4, 2])
    padded = array_ops.pad(image, paddings, constant_values=pad_color)

    padded_shape = [
        None if _is_tensor(i) else i
        for i in [batch, target_height, target_width, depth]
    ]
    padded.set_shape(padded_shape)

    if not is_batch:
      padded = array_ops.squeeze(padded, axis=[0])

    return padded

def zoom_image(image: Union[np.ndarray, tf.Tensor], zoom: float,
               pad_color: int = 0) -> Union[np.ndarray, tf.Tensor]:
    """ Image zoom with tensorflow functions.

    Arguments:
        image:  Either OpenCV (`np.ndarray`) or Tensorflow (`tf.Tensor`) image array.
            The image must be of supported `dtype` (`uint8` or `float`).
        zoom:   Zoom factor, float value greater than 0.
            If it is less than 1, then image is zoomed in (become larger),
            and if greater - zoomed out (become smaller) (this is oppozite to OpenCV's zoom behaviour).
        pad_color:  padding color (0 - black, 255 - white)

    Returns:
        zoomed image of the same type and size as original
    """

    if zoom < 0.0:
        raise ValueError(f'Zoom must be greater than zero')

    if zoom == 1.0:
        # No zoom
        return image

    is_tensor = isinstance(image, tf.Tensor)
    if not is_tensor:
        shape = image.shape
        image_size = np.array(shape[:2])
    else:
        shape = image.get_shape()
        image_size = np.array(shape[:2]) if (shape.ndims or 1) < 4 else np.array(shape[1:3])

    if zoom < 1.0:
        # Zoom in (result is larger)
        crop_image = tf.image.central_crop(image, zoom)
        zoomed_image = tf.image.resize(crop_image, image_size, preserve_aspect_ratio=False)
    else:
        # Zoom out (result is smaller)
        zoom_size = (zoom * image_size).astype('int')
        offset_height, offset_width = (zoom_size[0]-image_size[0]) // 2, (zoom_size[1]-image_size[1]) // 2
        target_height, target_width = zoom_size[0], zoom_size[1]
        resized_image = _pad_to_bounding_box_color(image, offset_height, offset_width, target_height, target_width, pad_color)
        zoomed_image = tf.image.resize(resized_image, image_size, preserve_aspect_ratio=False)

    zoomed_image = tf.cast(zoomed_image, image.dtype)
    return zoomed_image.numpy() if not is_tensor and isinstance(zoomed_image, tf.Tensor) else zoomed_image #type:ignore

def predict_image_files_zoom(
    model_base: ModelBase, 
    model: KerasModel,
    file_names: Iterable[str],
    true_labels: Iterable[str] | None = None,
    zoom_levels: Iterable[float] | None = None,
    show_processed_image: bool = False):
    """ Run a pretrained model prediction on multiple image files with optional zooming and display the result """

    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None
    class_names = model_base.get_class_labels()
    for file_name, true_label in zip_longest(file_names, true_labels or []):
        image = tf.image.decode_image(tf.io.read_file(file_name), channels=3)

        for zoom in zoom_levels or [1.0]:
            zoomed_image = zoom_image(image, zoom, 255)
            zoomed_image = tf.cast(zoomed_image, tf.uint8)

            predicted_label, predicted_prob, processed_image = predict_image(model_base, model, zoomed_image, return_image=True)

            plt.figure(f'{file_name} @ {zoom}')
            plt.title(f'{true_label or "?"} <- {predicted_label} ({predicted_prob:.2%}) @ {zoom}')
            img_to_show = model_base.restore_processed_image(processed_image.numpy()) if show_processed_image else zoomed_image.numpy()
            plt.imshow(img_to_show, cmap=cmap)
            plt.axis('off')

    plt.show()

def predict_image_file(
    model_base: ModelBase, 
    model: KerasModel,
    file_name: str,
    true_label: str = None,
    show_processed_image: bool = False):
    """ Run a pretrained model prediction on an image file and display the result """

    predict_image_files_zoom(model_base, model, [file_name], [true_label], show_processed_image=show_processed_image)

def predict_image_files(
    model_base: ModelBase, 
    model: KerasModel,
    file_names: Iterable[str],
    true_labels: Iterable[str] = None,
    show_processed_image: bool = False):
    """ Run a pretrained model prediction on multiple image files and display the result """

    predict_image_files_zoom(model_base, model, file_names, true_labels, show_processed_image=show_processed_image)

def show_prediction_samples(
    model_base: ModelBase, 
    model: KerasModel,
    tfds: tf.data.Dataset,
    num_samples: int = 9):
    """ Show prediction samples from test dataset """

    plt.figure(figsize=(8, 8))
    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None
    class_names = model_base.get_class_labels()
    for images, labels in get_dataset_samples(tfds):
        for i in range(min(num_samples, labels.shape[0])):
            _ = plt.subplot(int(num_samples/3), int(num_samples/3), i + 1)
            label = class_names[np.argmax(labels[i])]
            image = images[i].numpy()
            prediction = model.predict(np.array([image]))
            most_likely = np.argmax(prediction)
            predicted_label = class_names[most_likely]
            predicted_prob = prediction[0][most_likely]

            plt.title(f'{label} <- {predicted_label} ({predicted_prob:.2%})')
            plt.imshow(model_base.restore_processed_image(image), cmap=cmap)
            plt.axis('off')
    plt.show()

def filter_dataset_by_label(tfds: tf.data.Dataset, class_names: Iterable[str], label: str) -> tf.data.Dataset:
    """ Return a dataset subset containing images of given label only """
    label_index = class_names.index(label)
    return tfds.unbatch().filter(lambda _, label: tf.equal(tf.math.argmax(label), label_index)).batch(BATCH_SIZE)

