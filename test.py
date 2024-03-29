# LEGO sorter project
# Model testing script
# (c) lego-sorter team, 2022-2023

import os
import re
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from absl import app, flags

import lib.globals
from lib.model import load_model, make_model

from lib.image_dataset import (
    load_dataset, 
    show_prediction_samples,
    filter_dataset_by_label, 
    predict_image_files,
    predict_image_files_zoom,
    predict_dataset,
   
)

def get_file_label(file_name):
    _, short_name = os.path.split(file_name)
    matched = re.match('^\\d+', short_name)
    if matched and '_' in short_name:
        return short_name.split('_')[0]
    if matched:
        return matched.group(0)
    return None

def plot_confusion_matrix(actual, predicted, class_names):
    cm = tf.math.confusion_matrix(actual, predicted)
    ax = sns.heatmap(cm, annot=True, fmt='g')
    sns.set(rc={'figure.figsize':(12, 12)})
    sns.set(font_scale=1.4)
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Pred')
    ax.set_ylabel('True')
    labels = tf.unique(actual)[0].numpy()
    label_names = [class_names[int(i)] for i in labels]
    plt.xticks(labels, rotation=90)
    plt.yticks(labels, rotation=0)
    ax.xaxis.set_ticklabels(label_names)
    ax.yaxis.set_ticklabels(label_names)
    plt.show()

FLAGS = flags.FLAGS

del FLAGS.zoom_factor
flags.DEFINE_multi_float('zoom_factor', None, help='ROI zoom factor', short_name='zf')
flags.DEFINE_multi_string('file', None, 'Predict for image file(-s)', short_name='f')
flags.DEFINE_string('label', None, 'Predict for specific label', short_name='c')
flags.DEFINE_boolean('confusion', None, 'Plot confusion matrix', short_name='m')
flags.declare_key_flag('gray')
flags.declare_key_flag('edges')
flags.declare_key_flag('zoom')
flags.declare_key_flag('emboss')

def main(argv):
    """ Test the LEGO Sorter model """

    image_data = load_dataset()
    num_labels = len(image_data.class_names)
    model = make_model(num_labels)
    load_model(model)
    model.summary()

    if FLAGS.label:
        # prediction for given label
        label_data = filter_dataset_by_label(image_data.tfds, image_data.class_names, FLAGS.label)
        show_prediction_samples(model, label_data, image_data.class_names)

    elif FLAGS.file:
        # prediction for given file(s), optionally with zoom
        true_labels = [get_file_label(fn) for fn in FLAGS.file]
        if not FLAGS.zoom_factor:
            predict_image_files(model, FLAGS.file, image_data.class_names, true_labels)
        else:
            zoom_levels = FLAGS.zoom_factor
            if len(FLAGS.zoom_factor) == 1 and FLAGS.zoom_factor[0] == 0.0:
                zoom_levels = [x/100.0 for x in range(50, 300, 25)]
            predict_image_files_zoom(model, FLAGS.file, image_data.class_names, true_labels, zoom_levels, 
                show_processed_image=True)

    elif FLAGS.confusion:
        # plot confusion matrix
        true_labels, pred_labels = predict_dataset(model, image_data.tfds)
        plot_confusion_matrix(true_labels, pred_labels, image_data.class_names)
        
    else:
        # prediction for dataset sample
        show_prediction_samples(model, image_data.tfds, image_data.class_names)

if __name__ == '__main__':
    app.run(main)
