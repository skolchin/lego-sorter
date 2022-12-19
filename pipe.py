# LEGO sorter project
# Video recognition pipeline
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
flags.DEFINE_float('bbox_relax', 0.2, help='ROI bbox relax coefficient')
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

    blank_image = np.full(list(SCREEN_SIZE) + [3], imu.COLOR_BLACK, np.uint8)
    predict_image(model, blank_image, image_data.class_names)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        logger.error('Cannot open camera, exiting')
        return

    cam.set(cv2.CAP_PROP_FPS, FPS_RATE)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_SIZE[1])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_SIZE[0])

    frame_count = 0
    frame_caption = None
    frame_bbox = None
    back_sub = None
    video_out = None
    debug_show = False

    while True:
        ret, frame = cam.read()
        if not ret:
            logger.error('Cannot grab frame from camera, exiting')
            break

        if frame_count % FPS_RATE == 0:
            roi = None
            if back_sub is not None:
                bgmask = back_sub.apply(frame, learningRate=0)
                frame_bbox = bgmask_to_bbox(bgmask)
                if frame_bbox is not None:
                    roi = get_bbox_area(frame, frame_bbox, FLAGS.bbox_relax)
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    label, prob = predict_image(model, roi_rgb, image_data.class_names)
                    frame_caption = f'{label} ({prob:.2%})'

            if debug_show:
                show_hist_window(frame, roi)
                show_roi_window(roi, frame_caption)

        status_line = 0
        if back_sub is None:
            show_status(frame, 'Remove any objects and press B to capture background', important=True)
            status_line = 1
        show_status(frame, 'Press ESC or Q to quit, S for camera settings, C for video capture', line=status_line)

        if frame_bbox is not None:
            green_named_rect(frame, frame_bbox, frame_caption)

        frame_count += 1
        show_stream(frame)
        if video_out is not None:
            video_out.write(frame)

        key = int(cv2.waitKey(1) & 0xFF)
        match key:
            case 27 | 113:  # esc or q
                break

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

            case 100:    # d
                if debug_show: 
                    hide_hist_window()
                    hide_roi_window()
                debug_show = not debug_show

            case 115:   # s
                cam.set(cv2.CAP_PROP_SETTINGS, 1)

    if video_out is not None:
        video_out.release()
    cam.release()

if __name__ == '__main__':
    app.run(main)
