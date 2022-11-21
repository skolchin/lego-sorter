# LEGO sorter project
# Image loading and pre-processing functions (TF-dataset version)
# (c) kol, 2022

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
from typing import Tuple, Iterable

from model import BATCH_SIZE, IMAGE_SIZE, preprocess_input

IMAGE_DIR = os.path.join(os.path.split(__file__)[0], 'images')

def load_dataset(use_grayscale: bool = False) -> tf.data.Dataset:
    """ Loads images as TF dataset """
    ds = image_dataset_from_directory(
        IMAGE_DIR,
        label_mode='categorical',
        color_mode='grayscale' if use_grayscale else 'rgb',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        crop_to_aspect_ratio=True,
    )
    return ds

def split_dataset(ds: tf.data.Dataset, split_by: float=0.8) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """ Split TF dataset into train/test datasets """
    if BATCH_SIZE:
        train_size = int(len(ds.file_paths) * split_by / BATCH_SIZE)
        train_ds = ds.take(train_size)
        test_ds = ds.skip(train_size)
        return (train_ds, test_ds)
    else:
        train_size = int(len(ds.file_paths) * split_by)
        train_ds = ds.take(train_size)
        test_ds = ds.skip(train_size)
        return (train_ds, test_ds)

def show_samples(ds: tf.data.Dataset, num_samples: int = 9):
    """ Show random image samples from TF dataset """
    plt.figure(figsize=(8, 8))
    if BATCH_SIZE:
        for images, labels in ds.take(1):
            for i in range(num_samples):
                _ = plt.subplot(int(num_samples/3), int(num_samples/3), i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))
                label = np.argmax(labels[i])
                plt.title(ds.class_names[label])
                plt.axis("off")
    else:
        for i, (image, label) in enumerate(ds.take(num_samples)):
            _ = plt.subplot(int(num_samples/3), int(num_samples/3), i + 1)
            plt.imshow(image.numpy().astype("uint8"))
            plt.title(ds.class_names[np.argmax(label)])
            plt.axis("off")
    plt.show()

def predict_image(model, file_name: str, class_names: Iterable[str], 
    true_label: str = None, use_grayscale: bool = False):

    image = tf.keras.utils.load_img(file_name, target_size=IMAGE_SIZE, 
        interpolation='bicubic', keep_aspect_ratio=True)
    image = tf.keras.utils.img_to_array(image)
    if not use_grayscale:
        prepared_image = preprocess_input(np.array([image])).numpy()
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

def show_prediction_samples(model, ds: tf.data.Dataset, num_samples: int = 9, use_grayscale: bool = False):
    """ Show random prediction samples along with probability estimations """
    plt.figure(figsize=(8, 8))
    if BATCH_SIZE:
        for images, labels in ds.take(1):
            for i in range(num_samples):
                _ = plt.subplot(int(num_samples/3), int(num_samples/3), i + 1)
                label = ds.class_names[np.argmax(labels[i])]

                if use_grayscale:
                    image = tf.image.grayscale_to_rgb(images[i]).numpy()
                    prepared_image = preprocess_input(np.array([image]))
                    prepared_image = tf.image.rgb_to_grayscale(prepared_image).numpy()
                else:
                    image = images[i].numpy()
                    prepared_image = preprocess_input(np.array([image]))

                prediction = model.predict(prepared_image)
                most_likely = np.argmax(prediction)
                predicted_label = ds.class_names[most_likely]
                predicted_prob = prediction[0][most_likely]

                plt.title(f'{label} <- {predicted_label} ({predicted_prob:.2%})')
                plt.imshow(image.astype('uint8'))
                plt.axis("off")
        plt.show()

