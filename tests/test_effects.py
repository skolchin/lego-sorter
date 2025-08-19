# LEGO sorter project
# Verifies image display
# (c) kol, 2023

import os
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags
from lib.globals import OUTPUT_DIR
from matplotlib.widgets import Slider

logging.getLogger('lego-tracker').setLevel(logging.ERROR)

from lib.image_dataset import (
    fast_get_class_names, 
    _preprocess, 
    IMAGE_SIZE,
    zoom_image
)
from lib.custom_model import load_model, make_model

FLAGS = flags.FLAGS

def main(_):
    image = tf.image.decode_image(tf.io.read_file(os.path.join(OUTPUT_DIR, 'roi','3001_0127.png')), channels=3)
    resized_image = tf.keras.preprocessing.image.smart_resize(image, IMAGE_SIZE)

    class_names = fast_get_class_names()
    model = make_model(len(class_names))
    load_model(model)

    params = {
        'contrast': {
            'value': 1.0,
            'lower': -10.0,
            'upper': 10.0,
            'step': 1.0,
        },
        'brightness': {
            'value': 0.0,
            'lower': -30.0,
            'upper': 30.0,
            'step': 10.0,
        },
        'saturation': {
            'value': 0.0,
            'lower': 0.0,
            'upper': 30.0,
            'step': 1.0,
        },
        'zoom': {
            'value': 1.0,
            'lower': 1.0,
            'upper': 3.0,
            'step': 0.25,
        },
    }
    cmap = 'gray' if FLAGS.gray or FLAGS.edges else None

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.title('roi')
    plt.axis('off')

    def apply_adj():
        adj_image = resized_image
        adj_image = zoom_image(adj_image, params['zoom']['value'])
        adj_image = tf.image.adjust_contrast(adj_image, params['contrast']['value'])
        adj_image = tf.image.adjust_brightness(adj_image, params['brightness']['value'])
        adj_image = tf.image.adjust_saturation(adj_image, params['saturation']['value'])

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

    def make_update_func(key):
        def update_func(val):
            nonlocal params
            params[key]['value'] = val
            apply_adj()
        return update_func

    SLIDER_WIDTH = 0.0225
    SLIDER_SPACE = 0.10
    fig.subplots_adjust(left=(SLIDER_WIDTH+SLIDER_SPACE)*len(params), bottom=0.25)
    x = 0.1
    for key, par in params.items():
        par['slider'] = Slider(
            ax=fig.add_axes([x, 0.25, SLIDER_WIDTH, 0.63]),
            label=key,
            valmin=par['lower'],
            valmax=par['upper'],
            valstep=par['step'],
            valinit=par['value'],
            orientation="vertical"
        )
        par['slider'].on_changed(make_update_func(key))
        x += SLIDER_WIDTH+SLIDER_SPACE

    apply_adj()
    plt.show()

if __name__ == '__main__':
    app.run(main)
