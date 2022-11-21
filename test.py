# LEGO sorter project
# Model testing (TF-dataset version)
# (c) kol, 2022

import os
import click
import re
from keras.models import load_model
from model import MODEL_DIR

from image_dataset import (
    load_dataset, 
    show_prediction_samples,
    predict_image
)

def get_label(file_name):
    _, short_name = os.path.split(file_name)
    matched = re.match('^\\d+', short_name)
    if matched and '_' in short_name:
        return short_name.split('_')[0]
    if matched:
        return matched.match
    return None

@click.command()
@click.option('--image', '-f', '-i', 'file_name', type=click.Path(exists=True, dir_okay=False))
@click.option('--gray', '-g', '--grayscale', 'use_grayscale', is_flag=True, default=False, show_default=True,
    help='Convert images to grayscale')
def main(file_name: str, use_grayscale: bool):
    model = load_model(MODEL_DIR)
    model.layers[0].summary()
    model.summary()

    file_name = 'out\\3003_test.png'
    image_data = load_dataset(use_grayscale)
    if not file_name:
        show_prediction_samples(model, image_data, use_grayscale)
    else:
        true_label = get_label(file_name)
        predict_image(model, file_name, image_data.class_names, true_label, use_grayscale)

if __name__ == '__main__':
    main()
