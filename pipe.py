# LEGO sorter project
# Video recognition pipeline demo
# (c) kol, 2022

import os
import cv2
import logging
from absl import app, flags
from datetime import datetime

logging.basicConfig(format='%(levelname)s: %(message)s')
logger = logging.getLogger('lego-tracker')

from lib.globals import OUTPUT_DIR
from lib.pipe_utils import *
from lib.status_info import StatusInfo
from lib.object_tracker import track_detect
from lib.model import load_model, make_model
from lib.image_dataset import fast_get_class_names, predict_image, predict_image_probs

FLAGS = flags.FLAGS
flags.declare_key_flag('gray')
flags.declare_key_flag('edges')
flags.declare_key_flag('zoom')
flags.declare_key_flag('zoom_factor')
flags.declare_key_flag('brightness_factor')

flags.DEFINE_integer('camera', 0, short_name='c', help='Camera ID')
flags.DEFINE_string('file', None, short_name='f', help='Process video from given file')
flags.DEFINE_boolean('debug', False, help='Start with debug info displayed')
flags.DEFINE_boolean('video', False, help='Start with video capture')

HELP_INFO = 'Press ESC or Q to quit, S for camera settings, C for video capture'

def main(_):
    """ Video recognition pipeline """
    
    logger.setLevel(logging.INFO)
    show_welcome_screen()
    cv2.waitKey(10)

    class_names = fast_get_class_names()
    model = make_model(len(class_names))
    load_model(model)

    frame = np.full(list(FRAME_SIZE) + [3], imu.COLOR_BLACK, np.uint8)
    predict_image(model, frame, class_names)

    if FLAGS.file:
        logger.info(f'Processing video file {FLAGS.file}')
        cam = cv2.VideoCapture(FLAGS.file)
    else:
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            logger.error('Cannot open camera, exiting')
            return

        cam.set(cv2.CAP_PROP_FPS, FPS_RATE)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[1])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[0])

    ref_images = get_ref_images(class_names)

    frame_count = 0
    roi = None
    roi_caption = None
    roi_label = None
    video_out = None
    show_debug = FLAGS.debug
    show_preprocessed = False
    logger.setLevel(logging.DEBUG if show_debug else logging.INFO)

    status_info = StatusInfo(max_len=2)
    if not FLAGS.file:
        status_info.append(HELP_INFO)

    if FLAGS.video:
        fn = os.path.join(OUTPUT_DIR, f'pipe_{datetime.now().strftime("%y%m%d_%H%M%S")}.mp4')
        logger.info('Starting video output to %s', fn)
        video_out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, tuple(reversed(FRAME_SIZE)))

    for (frame, roi_bbox, detection) in track_detect(
        cam, 
        lambda roi: predict_image_probs(model, roi, class_names), 
        track_time=2.0,
        replace_bg_color=(255,255,255)):

        if detection is not None:
            roi, new_roi_label, roi_prob = detection
            if new_roi_label == roi_label:
                roi_caption = None
            else:
                roi_label = new_roi_label
                roi_caption = f'{roi_label} ({roi_prob:.2%})'
                status_info.append(f'Detection: {roi_caption}', True)
                logger.info(f'New detection: {roi_caption}')

        if show_preprocessed:
            frame = preprocess_image(frame)
        else:
            status_info.apply(frame)
            if roi_bbox is not None:
                red_named_rect(frame, roi_bbox, roi_caption)

        show_frame(frame)

        if video_out is not None:
            video_out.write(frame)

        if frame_count % 10 == 0 and show_debug:
            ref = None 
            if roi_label:
                ref = ref_images.get(roi_label)
                if show_preprocessed:
                    roi = preprocess_image(roi)
                    ref = preprocess_image(ref)
                show_roi_window(roi, roi_caption)
                show_ref_window(ref, roi_label)

            show_hist_window(frame, roi, ref, log_scale=True)

        frame_count += 1
        key = int(cv2.waitKey(1) & 0xFF)
        match key:
            case 27 | 113:  # esc or q
                break

            case 32:    # space
                status_info.assign_and_apply(frame, 'Paused, press SPACE to resume', True)
                show_frame(frame)
                while int(cv2.waitKey(10) & 0xFF) not in (27, 32, 113):
                    pass

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
                if not FLAGS.file:
                    cam.set(cv2.CAP_PROP_SETTINGS, 1)

    if video_out is not None:
        video_out.release()
    cam.release()

if __name__ == '__main__':
    app.run(main)
