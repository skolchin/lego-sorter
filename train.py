# LEGO sorter project
# Model training script
# (c) kol, 2022

import logging
import matplotlib.pyplot as plt
from absl import app, flags

logging.basicConfig(format='%(levelname)s: %(message)s')

import lib.globals
from lib.model import (
    make_model, 
    load_model, 
    get_checkpoint_callback, 
    get_early_stopping_callback,
    cleanup_checkpoints
)

from lib.image_dataset import (
    load_dataset, 
    show_samples,
    split_dataset,
    augment_dataset,
    show_prediction_samples
)

def plot_history(history):
    _, axs = plt.subplots(1, 2, sharey=True)
    for i, metric in enumerate([('categorical_accuracy', 'val_categorical_accuracy'), ('loss', 'val_loss')]):
        axs[i].plot(history.history[metric[0]])
        axs[i].plot(history.history[metric[1]])
        axs[i].set_title(f'{metric[0].capitalize()} plot')
        axs[i].legend(['train', 'val'], loc='upper left')
    plt.show()

FLAGS = flags.FLAGS
flags.DEFINE_integer('epoch', default=30, lower_bound=1, short_name='n',
    help='Number of epoch to train model for')
flags.DEFINE_boolean('show', default=True, help='Show image samples and plots')
flags.DEFINE_boolean('save', default=True, help='Save model to disk')
flags.declare_key_flag('gray')
flags.declare_key_flag('edges')
flags.declare_key_flag('zoom')

def main(argv):
    """ Train the LEGO Sorter model """

    image_data = load_dataset()
    if FLAGS.show:
        show_samples(image_data.tfds, image_data.class_names)
 
    num_labels = len(image_data.class_names)
    train_data, test_data = split_dataset(image_data)
    aug_data = augment_dataset(train_data)

    model = make_model(num_labels)
    print('\nModel summary:\n---')
    model.summary()
    load_model(model)

    if not FLAGS.save:
        callbacks_list = [get_early_stopping_callback()]
    else:
        callbacks_list = [get_early_stopping_callback(), get_checkpoint_callback()]

    history = model.fit(
        aug_data, 
        validation_data=test_data, 
        epochs=FLAGS.epoch,
        callbacks=callbacks_list)

    if FLAGS.save:
        cleanup_checkpoints()

    if FLAGS.show:
        plot_history(history)
        show_prediction_samples(model, test_data, image_data.class_names)

    # metrics = model.evaluate(test_data)
    # print(f'\nEvaluation metrics:\n---') 
    # print('\n'.join([f'{m}={v:.4%}' for m, v in zip(model.metrics_names, metrics)]))

if __name__ == '__main__':
    app.run(main)
