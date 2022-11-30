# LEGO sorter project
# Model testing (TF-dataset version)
# (c) kol, 2022

import os
import re
import tensorflow as tf
import matplotlib.pyplot as plt
from absl import app, flags

from model import load_model, make_model

from image_dataset import (
    load_dataset, 
    show_prediction_samples,
    split_dataset,
    filter_dataset_by_label, 
    predict_image,
    get_predictions
)

def get_file_label(file_name):
    _, short_name = os.path.split(file_name)
    matched = re.match('^\\d+', short_name)
    if matched and '_' in short_name:
        return short_name.split('_')[0]
    if matched:
        return matched.match
    return None

FLAGS = flags.FLAGS
flags.DEFINE_string('file', None, 'Predict for given image file', short_name='f')
flags.DEFINE_string('label', None, 'Predict for specific label', short_name='c')
flags.DEFINE_boolean('confusion', None, 'Plot confusion matrix', short_name='m')
flags.DEFINE_boolean('gray', False, 'Use grayscale-trained model', short_name='g')

def main(argv):
    """ Test the LEGO Sorter model """

    image_data = load_dataset(FLAGS.gray)
    num_labels = len(image_data.class_names)
    model = make_model(num_labels, FLAGS.gray)
    load_model(model)
    model.summary()

    if FLAGS.label:
        # prediction for given label
        label_data = filter_dataset_by_label(image_data.tfds, image_data.class_names, FLAGS.label)
        show_prediction_samples(model, label_data, image_data.class_names, use_grayscale=FLAGS.gray)

    elif FLAGS.file:
        # prediction for given file
        true_label = get_file_label(FLAGS.file)
        predict_image(model, FLAGS.file, image_data.class_names, true_label, FLAGS.gray)

    elif FLAGS.confusion:
        # plot confusion matrix
        _, test_data = split_dataset(image_data)
        predictions = get_predictions(model, test_data, image_data.class_names, use_grayscale=FLAGS.gray)
        true_labels, pred_labels = [p[0] for p in predictions], [p[1] for p in predictions]
        conf_matrix = tf.confusion_matrix(labels=true_labels, predictions=pred_labels)

        ax = plt.subplots(figsize=(8, 8))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions')
        plt.ylabel('Actuals')
        plt.title('Confusion Matrix')
        plt.show()
    else:
        # prediction for dataset sample
        show_prediction_samples(model, image_data.tfds, image_data.class_names, use_grayscale=FLAGS.gray)

if __name__ == '__main__':
    app.run(main)
