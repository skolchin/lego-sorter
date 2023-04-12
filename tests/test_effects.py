# LEGO sorter project
# Verifies image display
# (c) kol, 2023

import numpy as np
import logging
import tensorflow as tf
from absl import app, flags
from root_dir import ROOT_DIR, OUTPUT_DIR
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

logging.getLogger('lego-tracker').setLevel(logging.ERROR)

from lib.image_dataset import (
    fast_get_class_names, 
    _preprocess, 
    IMAGE_SIZE,
)
from lib.model import load_model, make_model

FLAGS = flags.FLAGS

def main(_):
    image = tf.image.decode_image(tf.io.read_file('out/roi/3001_0127.png'), channels=3)
    resized_image = tf.keras.preprocessing.image.smart_resize(image, IMAGE_SIZE)

    class_names = fast_get_class_names()
    model = make_model(len(class_names))
    load_model(model)

    contrast = 1.0
    brightness = 1.0
    saturation = 0.0
    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.title('roi')
    plt.axis('off')

    def apply_adj():
        adj_image = resized_image
        adj_image = tf.image.adjust_contrast(adj_image, contrast)
        adj_image = tf.image.adjust_brightness(adj_image, brightness)
        adj_image = tf.image.adjust_saturation(adj_image, saturation)

        prepared_image, _ = _preprocess(tf.expand_dims(adj_image, 0))
        image = prepared_image[0].numpy()
        ax.imshow(image.astype('uint8'), cmap=cmap)
    
        prepared_image = tf.keras.preprocessing.image.smart_resize(prepared_image, IMAGE_SIZE)
        prediction = model.predict(prepared_image, verbose=0)
        most_likely = np.argmax(prediction)
        label = class_names[most_likely]
        prob = prediction[0][most_likely]

        ax.set_title(f'{label} ({prob:.4f})')
        fig.canvas.draw_idle()

    def update_contrast(val):
        nonlocal contrast
        contrast = val
        apply_adj()

    def update_brightness(val):
        nonlocal brightness
        brightness = val
        apply_adj()

    def update_saturation(val):
        nonlocal saturation
        saturation = val
        apply_adj()

    fig.subplots_adjust(left=0.7, bottom=0.25)
    s1 = Slider(
        ax=fig.add_axes([0.1, 0.25, 0.0225, 0.63]),
        label='contrast',
        valmin=-10.0,
        valmax=10.0,
        valstep=1.0,
        valinit=contrast,
        orientation="vertical"
    )
    s1.on_changed(update_contrast)

    s2 = Slider(
        ax=fig.add_axes([0.3, 0.25, 0.0225, 0.63]),
        label='brightness',
        valmin=-30.0,
        valmax=30.0,
        valstep=10.0,
        valinit=brightness,
        orientation="vertical"
    )
    s2.on_changed(update_brightness)

    s3 = Slider(
        ax=fig.add_axes([0.5, 0.25, 0.0225, 0.63]),
        label='saturation',
        valmin=0.0,
        valmax=30.0,
        valstep=1,
        valinit=saturation,
        orientation="vertical"
    )
    s3.on_changed(update_saturation)

    plt.show()

if __name__ == '__main__':
    app.run(main)
