# LEGO sorter project
# Image loading and pre-processing functions (simple data processing) (WIP)
# (c) lego-sorter team, 2022-2023

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from typing import Tuple

from lib.model import preprocess_input

IMAGE_DIR = os.path.join(os.path.split(__file__)[0], 'images')

def load_image_data(max_files: int = None) -> pd.DataFrame:
    """ Load image list as a PD dataframe with columns ('image', 'filename', 'label') """
    labels = sorted([e.name for e in os.scandir(IMAGE_DIR) if e.is_dir()])
    images = []
    for label in labels:
        dir = os.path.join(IMAGE_DIR, label)
        files = sorted([e.name for e in os.scandir(dir) if e.is_file()])
        images.extend([(None, os.path.join(dir,fn), label) for fn in files])
        if max_files and len(images) > max_files: break

    return pd.DataFrame(images, columns=['image', 'filename', 'label'])

def split_data(df: pd.DataFrame, split_by: float=0.8) -> Tuple[Tuple, Tuple]:
    """ Split PD dataframe into train/test datasets """

    get_all_images(df)
    df_train = df.sample(frac=split_by)
    df_test = df.drop(df_train.index)
    labels = list(df['label'].unique())

    X_train = np.array([k for k in df_train['image']])
    y_train = to_categorical(np.array(df_train['label'].map(labels.index)))

    X_test = np.array([k for k in df_test['image']])
    y_test = to_categorical(np.array(df_test['label'].map(labels.index)))

    X_test = preprocess_input(X_test / 255.0)
    X_train = preprocess_input(X_train / 255.0)

    return ((X_train, y_train), (X_test, y_test))

def get_image(df: pd.DataFrame, index):
    """ Get an image for particular row """
    image = df.iloc[index]['image']
    if image is None:
        image = cv2.imread(df.iloc[index]['filename'])
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        df.iloc[index]['image'] = image
    return image

def get_all_images(df: pd.DataFrame):
    """ Get an image from particular row """
    for row in df.itertuples(index=True):
        get_image(df, row.Index)
    return df['image']

def show_data_samples(df: pd.DataFrame, num_samples: int = 9):
    """ Show random image samples from PD dataframe """
    plt.figure(figsize=(8, 8))
    for idx, row in enumerate(df.sample(num_samples).itertuples()):
        ax = plt.subplot(int(num_samples/3), int(num_samples/3), idx+1)
        plt.imshow(get_image(df, row.Index).astype("uint8"))
        plt.title(row.label)
        plt.axis("off")
    plt.show()

def show_prediction_samples(model, df: pd.DataFrame, num_samples: int = 9):
    labels = list(df['label'].unique())
    plt.figure(figsize=(10, 10))
    for i, row in enumerate(df.sample(num_samples).itertuples(index=True)):
        ax = plt.subplot(3, 3, i + 1)
        image = get_image(df, row.Index)

        prediction = model.predict(preprocess_input(np.array([image])))
        most_likely = np.argmax(prediction)
        predicted_label = labels[most_likely]
        predicted_prob = prediction[0][most_likely]

        plt.title(f'{row.label} <- {predicted_label} ({predicted_prob:.2%})')
        plt.imshow(image)
        plt.axis("off")
    plt.show()

# ds = load_image_dataset()
# print(f'Number of images: {len(ds.file_paths)}')
# show_dataset_samples_unbatched(ds)

# train, test = split_dataset(ds)
# print(train, test)

# df = load_image_data(100)
# print(f'Number of images: {df.shape[0]}')
# get_image(df, 1)
# get_image(df, 1)
# show_data_samples(df)

# df = load_image_data(10)
# (X_train, y_train), (X_test, y_test) = split_data(df)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

