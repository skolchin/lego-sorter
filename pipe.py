# LEGO sorter project
# Recognition pipeline
# (c) kol, 2022

import os
import cv2
import logging
from absl import app, flags
from datetime import datetime

logging.basicConfig(format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import lib.globals
from lib.pipe_utils import *

from lib.model import load_model, make_model
from lib.image_dataset import load_dataset,  predict_image

FLAGS = flags.FLAGS
flags.declare_key_flag('gray')
flags.declare_key_flag('edges')

def main(argv):
    """ Video recognition pipeline """
    show_welcome_screen()
    cv2.waitKey(10)

    image_data = load_dataset()
    num_labels = len(image_data.class_names)
    model = make_model(num_labels)
    load_model(model)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        logger.error('Cannot open camera, exiting')
        return

    cam.set(cv2.CAP_PROP_FPS, 30.0)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_SIZE[1])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_SIZE[0])

    back_sub = None
    frame_count = 0
    frame_caption = '?'
    video_out = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        status_line = 0
        if back_sub is None:
            show_status(frame, 'Remove any objects and press B to capture background', important=True)
            status_line = 1
        show_status(frame, 'Press ESC or Q to quit, S for video settings, C for video capture', line=status_line)

        if back_sub is not None:
            bgmask = back_sub.apply(frame, learningRate=0)
            bbox = bgmask_to_bbox(bgmask)
            if bbox is not None:
                if frame_count % 30 == 0:
                    pred_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pred_image = get_bbox_area(pred_image, bbox, 0.2)
                    pred_label, pred_prob = predict_image(model, pred_image, image_data.class_names)
                    frame_caption = f'{pred_label} ({pred_prob:.2%})'

                frame_count += 1
                green_named_rect(frame, frame_caption, bbox)

        show_stream(frame)
        if video_out is not None:
            video_out.write(frame)

        key = int(cv2.waitKey(30) & 0xFF)
        match key:
            case 27 | 113:  # esc or q
                break

            case 115:   # s
                cam.set(cv2.CAP_PROP_SETTINGS, 1)

            case 98:    # b
                if back_sub is None:
                    back_sub = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=10, detectShadows=True)
                    back_sub.apply(frame, learningRate=-1)
                else:
                    back_sub = None

            case 99:    # c
                if video_out is None:
                    fn = os.path.join('out', f'pipe_{datetime.now().strftime("%y%m%d_%H%M%S")}.mp4')
                    logger.info('Starting video output to %s', fn)
                    video_out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, tuple(reversed(SCREEN_SIZE)))
                else:
                    video_out.release()
                    video_out = None
                    logger.info('Video output stopped')

    if video_out is not None:
        video_out.release()
    cam.release()

if __name__ == '__main__':
    app.run(main)
