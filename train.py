# LEGO sorter project
# Model training (TF-dataset version)
# (c) kol, 2022

import click
import matplotlib.pyplot as plt
from model import make_model, load_model, checkpoint_callback, CHECKPOINT_DIR

from image_dataset import (
    load_dataset, 
    show_samples,
    split_dataset, 
    show_prediction_samples
)

def plot_history(history):
    _, axs = plt.subplots(1, 2, sharey=True)
    for i, metric in enumerate([('accuracy', 'val_accuracy'), ('loss', 'val_loss')]):
        axs[i].plot(history.history[metric[0]])
        axs[i].plot(history.history[metric[1]])
        axs[i].set_title(f'{metric[0].capitalize()} plot')
        axs[i].legend(['train', 'val'], loc='upper left')
    plt.show()

@click.command()
@click.option('--epoch', '-n', 'num_epoch', type=int, default=30, show_default=True,
    help='Number of epoch to train model for')
@click.option('--show/--no-show', 'show_images', is_flag=True, default=True, show_default=True,
    help='Show image samples and plots')
@click.option('--save/--no-save', 'save_model', is_flag=True, default=True, show_default=True,
    help='Save model to disk')
@click.option('--gray', '-g', '--grayscale', 'use_grayscale', is_flag=True, default=False, show_default=True,
    help='Convert images to grayscale')
def main(show_images: bool, save_model: bool, num_epoch: int, use_grayscale: bool):
    """ Train the LEGO Sorter model """

    image_data = load_dataset(use_grayscale=use_grayscale)
    if show_images:
        show_samples(image_data)

    num_labels = len(image_data.class_names)
    train_data, test_data = split_dataset(image_data)

    model = make_model(num_labels, use_grayscale=use_grayscale)
    print('\nModel summary:\n---')
    model.summary()

    if not save_model:
        callbacks_list = []
    else:
        callbacks_list = [checkpoint_callback()]
        load_model(model)

    history = model.fit(train_data, validation_data=test_data, epochs=num_epoch,
        callbacks=callbacks_list)

    if show_images:
        plot_history(history)

    # metrics = model.evaluate(test_data)
    # print(f'\nEvaluation metrics:\n---') 
    # print('\n'.join([f'{m}={v:.4%}' for m, v in zip(model.metrics_names, metrics)]))

    if show_images:
        show_prediction_samples(model, image_data, use_grayscale=use_grayscale)

if __name__ == '__main__':
    main()
