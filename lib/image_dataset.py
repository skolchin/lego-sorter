# LEGO sorter project
# Image loading and processing functions
# (c) lego-sorter team, 2022-2023

import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from absl import flags
from keras.utils import image_dataset_from_directory
from dataclasses import dataclass
from functools import partial
from itertools import zip_longest
from typing import Tuple, Iterable, Union

from lib.globals import IMAGE_DIR, BATCH_SIZE, IMAGE_SIZE
from lib.model import preprocess_input

FLAGS = flags.FLAGS
flags.DEFINE_float('zoom_factor', 0.3, help='Maximum zoom level for image augmentation (for --zoom model only)')
flags.DEFINE_float('brightness_factor', 0.3, help='Maximum brightness level for image augmentation')
flags.DEFINE_float('rotation_factor', 0.45, help='Maximum rotation in image augmentation')
flags.DEFINE_float('crop_factor', 0.0, upper_bound=1.0, help='Crop image factor (from center)')
flags.DEFINE_float('edge_emboss', 0.5, lower_bound=0.0, help='Edge embossing factor (for --emboss model only)')
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

def _preprocess(images, labels=None, seed=None):
    """ Image preprocessing function """
    if FLAGS.crop_factor:
        images = tf.image.central_crop(images, FLAGS.crop_factor)
        images = tf.image.resize(images, IMAGE_SIZE, 'bicubic', preserve_aspect_ratio=False)

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
        shape=(num_samples,), seed=seed, minval=FLAGS.rotation_factor*100.0, maxval=FLAGS.rotation_factor*100.0
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
    """ Load images into a dataset """

    seed = np.random.randint(1e6)
    ds = image_dataset_from_directory(
        IMAGE_DIR,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=None,
        image_size=IMAGE_SIZE,
        interpolation='bicubic',
        shuffle=False,
        crop_to_aspect_ratio=True,
        seed=seed,
    )
    num_files = len(ds.file_paths)
    buf_size = int(FLAGS.shuffle_buffer_size * num_files) if FLAGS.shuffle_buffer_size < 1 \
        else int(FLAGS.shuffle_buffer_size)
    return ImageDataset(
        ds.shuffle(buffer_size=buf_size, reshuffle_each_iteration=False, seed=seed) \
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
    class_names: Iterable[str],
    return_image: bool = False) -> Tuple[str, float]:
    """ Run a pretrained model prediction on an image """

    resized_image = tf.keras.preprocessing.image.smart_resize(image, IMAGE_SIZE)
    prepared_image, _ = _preprocess(tf.expand_dims(resized_image, 0))

    prediction = model.predict(prepared_image, verbose=0)
    most_likely = np.argmax(prediction)
    label = class_names[most_likely]
    prob = prediction[0][most_likely]

    if not return_image:
        return label, prob
    
    return label, prob, prepared_image[0]

def predict_image_probs(
    model: tf.keras.Model, 
    image: np.ndarray, 
    class_names: Iterable[str]) -> Iterable[Tuple[str, float]]:
    """ Run a pretrained model prediction on an image returning all detections and probabilities """

    resized_image = tf.keras.preprocessing.image.smart_resize(image, IMAGE_SIZE)
    prepared_image, _ = _preprocess(tf.expand_dims(resized_image, 0))

    prediction = model.predict(prepared_image, verbose=0)
    probs = []
    for prob, pred in sorted(zip(prediction[0], range(len(prediction[0]))), key=lambda x: x[0], reverse=True):
        label = class_names[pred]
        probs.append((label, prob))
    return probs

def predict_dataset(model: tf.keras.Model, tfds: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor]:
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
        array_ops.stack([
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
    shape = image.shape if not is_tensor else image.get_shape()
    image_size = np.array(shape[:2])
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
    return zoomed_image.numpy() if not is_tensor and isinstance(zoomed_image, tf.Tensor) else zoomed_image

def predict_image_files_zoom(
    model: tf.keras.Model, 
    file_names: Iterable[str], 
    class_names: Iterable[str], 
    true_labels: Iterable[str] = None,
    zoom_levels: Iterable[float] = None,
    show_processed_image: bool = False):
    """ Run a pretrained model prediction on multiple image files with optional zooming and display the result """

    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None
    for file_name, true_label in zip_longest(file_names, true_labels or []):
        image = tf.image.decode_image(tf.io.read_file(file_name), channels=3)

        for zoom in zoom_levels or [1.0]:
            zoomed_image = zoom_image(image, zoom, 255)
            zoomed_image = tf.cast(zoomed_image, tf.uint8)

            predicted_label, predicted_prob, processed_image = predict_image(model, zoomed_image, class_names, return_image=True)

            plt.figure(f'{file_name} @ {zoom}')
            plt.title(f'{true_label or "?"} <- {predicted_label} ({predicted_prob:.2%}) @ {zoom}')
            plt.imshow(processed_image.numpy() if show_processed_image else zoomed_image.numpy(), cmap=cmap)
            plt.axis('off')

    plt.show()

def predict_image_file(
    model: tf.keras.Model, 
    file_name: str, 
    class_names: Iterable[str], 
    true_label: str = None,
    show_processed_image: bool = False):
    """ Run a pretrained model prediction on an image file and display the result """

    predict_image_files_zoom(model, [file_name], class_names, [true_label], show_processed_image=show_processed_image)

def predict_image_files(
    model: tf.keras.Model, 
    file_names: Iterable[str], 
    class_names: Iterable[str], 
    true_labels: Iterable[str] = None,
    show_processed_image: bool = False):
    """ Run a pretrained model prediction on multiple image files and display the result """

    predict_image_files_zoom(model, file_names, class_names, true_labels, show_processed_image=show_processed_image)

def show_prediction_samples(
    model: tf.keras.Model, 
    tfds: tf.data.Dataset,
    class_names: Iterable[str],
    num_samples: int = 9):
    """ Show prediction samples from test dataset """
    
    plt.figure(figsize=(8, 8))
    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None
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
            plt.imshow(image.astype('uint8'), cmap=cmap)
            plt.axis('off')
    plt.show()

def filter_dataset_by_label(tfds: tf.data.Dataset, class_names: Iterable[str], label: str) -> tf.data.Dataset:
    """ Return a dataset subset containing images of given label only """
    label_index = class_names.index(label)
    return tfds.unbatch().filter(lambda _, label: tf.equal(tf.math.argmax(label), label_index)).batch(BATCH_SIZE)

def fast_get_class_names() -> Iterable[str]:
    """ Faster way of getting image class names """
    return [f for f in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, f))]

