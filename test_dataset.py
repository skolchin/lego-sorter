# LEGO sorter project
# TF dataset test
# (c) kol, 2022

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from image_dataset import (
    load_dataset, 
    show_samples,
    split_dataset, 
)

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

def main():
    image_data = load_dataset()
    show_samples(image_data)

    train_data, test_data = split_dataset(image_data)

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
    main()