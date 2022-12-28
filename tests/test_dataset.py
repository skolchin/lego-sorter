# LEGO sorter project
# TF dataset test
# (c) kol, 2022

import numpy as np
import pandas as pd
from absl import app, flags
from time import time
from root_dir import ROOT_DIR

import lib.globals
from lib.image_dataset import (
    load_dataset, 
    show_samples,
    split_dataset, 
    augment_dataset,
    get_dataset_samples,
    fast_get_class_names,
)

flags.declare_key_flag('gray')
flags.declare_key_flag('edges')
flags.declare_key_flag('zoom')

def get_labels(tag, tfds, class_names):
    real_labels = []
    for num_batch, (images, labels) in tfds.enumerate():
        batch_labels = [(np.argmax(label), class_names[np.argmax(label)]) for label in labels]
        for num_label, (label, class_name) in enumerate(batch_labels):
            real_labels.append({
                'x': num_batch.numpy(), 
                'n': num_label, 
                'tag': tag, 
                'y':  label, 
                'label': class_name
            })
    return real_labels

def measure_time(func: callable, *args, **kwargs):
    tm = time()
    ret = func(*args, **kwargs)
    print(f'Function {repr(func).split(" ")[1]} call took {time() - tm:.2f} seconds')
    return ret

def run_dataset_enum(tfds):
    for images, labels in get_dataset_samples(tfds, 5):
        for _ in zip(images, labels): pass

def main(argv):
    image_data = measure_time(load_dataset)
    class_names = fast_get_class_names()

    if image_data.class_names != class_names:
        print(f'Class names are different:\n\tdataset: {image_data.class_names}\n\tfast: {class_names}')
    classes_diff = set(image_data.class_names).symmetric_difference(class_names)
    if classes_diff:
        print(f'Classes non matched: {classes_diff}')

    train_data, test_data = measure_time(split_dataset,image_data)
    aug_data = measure_time(augment_dataset, test_data)
    measure_time(run_dataset_enum, aug_data)

    show_samples(aug_data, image_data.class_names)

    real_labels = \
        get_labels('train', train_data, image_data.class_names) + \
        get_labels('test', test_data, image_data.class_names)

    df = pd.DataFrame(real_labels)
    print('Labels sample:\n---')
    print(df.sample(5))

    train_labels = df[df.tag == 'train'].label.unique()
    test_labels = df[df.tag == 'test'].label.unique()
    label_diff = set(test_labels).difference(train_labels)
    if label_diff:
        print(f'Labels only in test: {label_diff}')
        
    label_diff = set(train_labels).difference(test_labels)
    if label_diff:
        print(f'Labels only in train: {label_diff}')

    group_var = pd.DataFrame(df.groupby(['tag', 'x'])['y'].var())
    group_var_mean = group_var.groupby('tag').mean()
    print('\nVariance mean:\n---')
    print(group_var_mean)

if __name__ == '__main__':
    app.run(main)
