# LEGO sorter project
# Determines average bounding box size of source images
# (c) kol, 2023

import cv2
import logging
import pandas as pd
from absl import app
from pathlib import Path
from tqdm import tqdm
from lib.globals import ROOT_DIR, OUTPUT_DIR

logging.getLogger('lego-tracker').setLevel(logging.ERROR)

from lib.pipe_utils import bgmask_to_bbox, show_frame, green_named_rect
from lib.object_tracker import track_detect
from lib.image_dataset import fast_get_class_names, predict_image, predict_image_probs
from lib.custom_model import load_model, make_model

from test_static_pipe import find_bgmask_multichannel, find_bgmask_thresh

def main(_):
    data = []
    files = [fn for fn in Path(ROOT_DIR).joinpath('images').glob('**/*.png') \
        if not fn.is_dir() and not fn.parent.name.startswith('images') ]
    for fn in tqdm(files, desc='Processing static images'):
        label = fn.parent.name
        img = cv2.imread(str(fn))
        bgmask = find_bgmask_multichannel(img)
        bbox = bgmask_to_bbox(bgmask)
        if bbox:
            data.append((label, bbox[2], bbox[3]))

    df_img = pd.DataFrame(data, columns=['label', 'w', 'h'])
    print(f'IMAGES: overall average width / heigth: {df_img.w.mean()} / {df_img.h.mean()}')
    print(f'IMAGES: average width / heigth per class:\n {df_img.groupby("label").mean()}')

    class_names = fast_get_class_names()
    model = make_model(len(class_names))
    load_model(model)

    data = []
    cam = cv2.VideoCapture(str(Path(OUTPUT_DIR).joinpath('2023-03-26 16-51-31.mkv')))
    for (frame, bbox, detection) in tqdm(track_detect(cam, lambda roi: predict_image_probs(model, roi, class_names), track_time=2.0),
                                            desc='Processing video file'):
        if detection:
            data.append((detection[1], bbox[2], bbox[3]))

    df_video = pd.DataFrame(data, columns=['label', 'w', 'h'])
    print(f'VIDEO: overall average width / heigth: {df_video.w.mean()} / {df_video.h.mean()}')
    print(f'VIDEO: average width / heigth per class:\n {df_video.groupby("label").mean()}')

    df_merged['w_ratio'] = df_merged['w_video'] / df_merged['w_image']
    df_merged['h_ratio'] = df_merged['h_video'] / df_merged['h_image']

    df_merged = pd.merge(df_video, df_img, how='inner', on='label', suffixes=['_video', '_image'])
    print(f'\nTOTAL: average width / heigth per class:\n {df_merged.groupby("label").mean()}')


if __name__ == '__main__':
    app.run(main)
