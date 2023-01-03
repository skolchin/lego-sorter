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

from lib.globals import OUTPUT_DIR
from lib.pipe_utils import *
from lib.pipe_status import StatusInfo
from lib.model import load_model, make_model
from lib.image_dataset import fast_get_class_names, predict_image, predict_image_probs

FLAGS = flags.FLAGS
flags.DEFINE_boolean('equalize_luminosity', True, 
    help='Apply luminosity equalization filter', short_name='eq')
flags.DEFINE_float('zoom_level', 2.0, lower_bound=0.0,
    help='Zoom level', short_name='zl')
flags.declare_key_flag('gray')
flags.declare_key_flag('edges')
flags.declare_key_flag('zoom')

HELP_CAPTURE = 'Remove any objects and press B to capture background'
HELP_INFO = 'Press ESC or Q to quit, S for camera settings, C for video capture'

def main(argv):
    """ Video recognition pipeline """
    logger.setLevel(logging.INFO)
    show_welcome_screen()
    cv2.waitKey(10)

    class_names = fast_get_class_names()
    model = make_model(len(class_names))
    load_model(model)

    frame = np.full(list(FRAME_SIZE) + [3], imu.COLOR_BLACK, np.uint8)
    predict_image(model, frame, class_names)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        logger.error('Cannot open camera, exiting')
        return

    cam.set(cv2.CAP_PROP_FPS, FPS_RATE)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[1])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[0])

    frame_count = 0
    roi = None
    ref = None
    roi_label = None
    roi_prob = None
    roi_caption = None
    roi_bbox = None
    back_sub = None
    video_out = None
    show_debug = False
    show_preprocessed = False
    eq_filter = imu.EqualizeLuminosity()

    status_info = StatusInfo()
    status_info.append(HELP_CAPTURE, important=True)
    status_info.append(HELP_INFO)

    while True:
        ret, frame = cam.read()
        if not ret:
            logger.error('Cannot grab frame from camera, exiting')
            break

        if frame_count % FPS_RATE == 0 and back_sub is not None:
            eq_frame = eq_filter(frame) if FLAGS.equalize_luminosity else frame
            bgmask = back_sub.apply(eq_frame, learningRate=0)
            roi_bbox = bgmask_to_bbox(bgmask)
            if roi_bbox is not None:
                roi = extract_roi(frame, roi_bbox, zoom_level=FLAGS.zoom_level)
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                labels_probs = predict_image_probs(model, roi_rgb, class_names)
                logger.debug(f'Top-3 detections: {labels_probs[:3]}')
                roi_label, roi_prob = labels_probs[0]
                roi_caption = f'{roi_label} ({roi_prob:.2%})'
                ref = get_ref_image(roi_label)
                if show_preprocessed:
                    roi = preprocess_image(roi)
                    ref = preprocess_image(ref)

        if show_preprocessed:
            frame = preprocess_image(frame)
        else:
            status_info.apply(frame)
            if roi_bbox is not None:
                green_named_rect(frame, roi_bbox, roi_caption)

        show_frame(frame)

        if frame_count % FPS_RATE == 0 and show_debug:
            show_hist_window(frame, roi, ref, log_scale=True)
            show_roi_window(roi, roi_caption)
            show_ref_window(ref, roi_label)

        if video_out is not None:
            video_out.write(frame)

        frame_count += 1
        key = int(cv2.waitKey(1) & 0xFF)
        match key:
            case 27 | 113:  # esc or q
                break

            case 98:    # b
                if back_sub is None:
                    eq_frame = eq_filter(frame) if FLAGS.equalize_luminosity else frame
                    back_sub = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=10, detectShadows=True)
                    back_sub.apply(eq_frame, learningRate=-1)
                    del status_info[0]
                else:
                    back_sub = None
                    roi_bbox = None
                    roi = None
                    ref = None
                    show_preprocessed = False
                    status_info.insert(0, HELP_CAPTURE, important=True)

            case 99:    # c
                if video_out is None:
                    fn = os.path.join(OUTPUT_DIR, f'pipe_{datetime.now().strftime("%y%m%d_%H%M%S")}.mp4')
                    logger.info('Starting video output to %s', fn)
                    video_out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, tuple(reversed(FRAME_SIZE)))
                else:
                    video_out.release()
                    video_out = None
                    logger.info('Video output stopped')

            case 100:    # d
                if show_debug: 
                    hide_hist_window()
                    hide_roi_window()
                    hide_ref_window()
                show_debug = not show_debug
                logger.setLevel(logging.DEBUG if show_debug else logging.INFO)

            case 112:   # p
                show_preprocessed = not show_preprocessed

            case 115:   # s
                cam.set(cv2.CAP_PROP_SETTINGS, 1)

    if video_out is not None:
        video_out.release()
    cam.release()

if __name__ == '__main__':
    app.run(main)
