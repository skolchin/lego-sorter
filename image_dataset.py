# LEGO sorter project
# Image loading and pre-processing functions (TF-dataset version)
# (c) kol, 2022

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
from collections import namedtuple
from typing import Tuple, Iterable, Union

from model import BATCH_SIZE, IMAGE_SIZE, preprocess_input

IMAGE_DIR = os.path.join(os.path.split(__file__)[0], 'images')

ImageDataset = namedtuple('ImageDataset', ['tfds', 'num_files', 'class_names'])

def load_dataset(use_grayscale: bool = False) -> ImageDataset:
    """ Load images as TF dataset """
    seed = np.random.randint(1e6)
    ds = image_dataset_from_directory(
        IMAGE_DIR,
        label_mode='categorical',
        color_mode='grayscale' if use_grayscale else 'rgb',
        batch_size=None,
        image_size=IMAGE_SIZE,
        shuffle=False,
        crop_to_aspect_ratio=False,
        seed=seed,
    )
    num_files = len(ds.file_paths)
    return ImageDataset(
        ds.shuffle(buffer_size=int(num_files/4), reshuffle_each_iteration=False, seed=seed).batch(BATCH_SIZE), 
        num_files,
        ds.class_names.copy()
    )

def split_dataset(ids: ImageDataset, split_by: float=0.8) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """ Split TF dataset into train/test subsets """
    train_size = int(ids.num_files * split_by / BATCH_SIZE)
    train_ds = ids.tfds.take(train_size)
    test_ds = ids.tfds.skip(train_size)
    return (train_ds, test_ds)

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
    for images, labels in get_dataset_samples(tfds):
        for i in range(num_samples):
            _ = plt.subplot(int(num_samples/3), int(num_samples/3), i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            label = np.argmax(labels[i])
            plt.title(class_names[label])
            plt.axis('off')
    plt.show()

def predict_image(
    model: tf.keras.Model, 
    file_name: str, 
    class_names: Iterable[str], 
    true_label: str = None, 
    use_grayscale: bool = False):
    """ Run a pretrained model prediction on given image file """

    image = tf.keras.utils.load_img(file_name, target_size=IMAGE_SIZE, 
                                    interpolation='bicubic', keep_aspect_ratio=False)
    image = tf.keras.utils.img_to_array(image)
    if not use_grayscale:
        prepared_image = preprocess_input(np.array([image]))
    else:
        prepared_image = preprocess_input(np.array([image]))
        prepared_image = tf.image.rgb_to_grayscale(prepared_image).numpy()

    prediction = model.predict(prepared_image)
    most_likely = np.argmax(prediction)
    predicted_label = class_names[most_likely]
    predicted_prob = prediction[0][most_likely]

    plt.title(f'{true_label or "?"} <- {predicted_label} ({predicted_prob:.2%})')
    plt.imshow(image.astype('uint8'))
    plt.axis('off')
    plt.show()

def show_prediction_samples(
    model: tf.keras.Model, 
    tfds: tf.data.Dataset,
    class_names: Iterable[str],
    num_samples: int = 9, 
    use_grayscale: bool = False):
    """ Show prediction samples from test dataset """
    
    plt.figure(figsize=(8, 8))
    for images, labels in get_dataset_samples(tfds):
        for i in range(num_samples):
            _ = plt.subplot(int(num_samples/3), int(num_samples/3), i + 1)
            label = class_names[np.argmax(labels[i])]

            if use_grayscale:
                image = tf.image.grayscale_to_rgb(images[i]).numpy()
                prepared_image = preprocess_input(np.array([image]))
                prepared_image = tf.image.rgb_to_grayscale(prepared_image).numpy()
            else:
                image = images[i].numpy()
                prepared_image = preprocess_input(np.array([image]))

            prediction = model.predict(prepared_image)
            most_likely = np.argmax(prediction)
            predicted_label = class_names[most_likely]
            predicted_prob = prediction[0][most_likely]

            plt.title(f'{label} <- {predicted_label} ({predicted_prob:.2%})')
            plt.imshow(image.astype('uint8'))
            plt.axis('off')
    plt.show()

def filter_dataset_by_label(tfds: tf.data.Dataset, class_names: Iterable[str], label: str) -> tf.data.Dataset:
    """ Return a dataset subset containing images of given label only """
    label_index = class_names.index(label)
    return tfds.unbatch().filter(lambda _, label: tf.equal(tf.math.argmax(label), label_index)).batch(BATCH_SIZE)

def get_predictions(
    model: tf.keras.Model, 
    tfds: tf.data.Dataset,
    class_names: Iterable[str],
    use_grayscale: bool = False,
    return_images: bool = False) -> Union[np.array, Tuple[np.array, np.array]]:
    """ Get predictions for dataset """

    ret_pred = []
    ret_images = []
    for images, labels in tfds:
        if use_grayscale:
            images = tf.image.grayscale_to_rgb(images)
            prepared_images = preprocess_input(images.numpy())
            prepared_images = tf.image.rgb_to_grayscale(prepared_images).numpy()
        else:
            prepared_images = preprocess_input(images.numpy())

        predictions = model.predict(prepared_images)
        for image, label, prediction in zip(images, labels, predictions):
            most_likely = np.argmax(prediction)
            predicted_label = class_names[most_likely]
            predicted_prob = prediction[0][most_likely]
            ret_pred.append([label, predicted_label, predicted_prob])
            if return_images:
                ret_images.append(image)

    return (np.array(ret_pred), np.array(ret_images)) if return_images else np.array(ret_pred)
