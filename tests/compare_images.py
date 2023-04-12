# LEGO sorter project
# Verifies image display
# (c) kol, 2023

import logging
import tensorflow as tf
from absl import app, flags
from root_dir import ROOT_DIR, OUTPUT_DIR
import matplotlib.pyplot as plt

logging.getLogger('lego-tracker').setLevel(logging.ERROR)

from lib.image_dataset import (
    fast_get_class_names, 
    load_dataset, 
    filter_dataset_by_label, 
    _preprocess, 
    IMAGE_SIZE,
)

FLAGS = flags.FLAGS

def main(_):
    ds = load_dataset()
    class_names = fast_get_class_names()
    ds_filtered = filter_dataset_by_label(ds.tfds, class_names, '3002')

    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None
    plt.figure(figsize=(8, 4))
    for images, labels in ds_filtered.take(1):
        _ = plt.subplot(131)
        plt.title('dataset')
        plt.axis('off')
        image = images[0].numpy()
        plt.imshow(image.astype('uint8'), cmap=cmap)
        break

    image = tf.image.decode_image(tf.io.read_file("images/3002/3002_288_288R.png"), channels=3)
    resized_image = tf.keras.preprocessing.image.smart_resize(image, IMAGE_SIZE)
    prepared_image, _ = _preprocess(tf.expand_dims(resized_image, 0))

    _ = plt.subplot(132)
    plt.title('source')
    plt.axis('off')
    image = prepared_image[0].numpy()
    plt.imshow(image.astype('uint8'), cmap=cmap)

    image = tf.image.decode_image(tf.io.read_file('out/roi/3001_0127.png'), channels=3)
    resized_image = tf.keras.preprocessing.image.smart_resize(image, IMAGE_SIZE)
    prepared_image, _ = _preprocess(tf.expand_dims(resized_image, 0))

    _ = plt.subplot(133)
    plt.title('roi')
    plt.axis('off')
    image = prepared_image[0].numpy()
    plt.imshow(image.astype('uint8'), cmap=cmap)

    plt.show()

if __name__ == '__main__':
    app.run(main)

