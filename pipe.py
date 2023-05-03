# LEGO sorter project
# Video recognition pipeline demo
# (c) lego-sorter team, 2022-2023

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

del FLAGS.zoom_factor
flags.DEFINE_float('zoom_factor', 2.5, short_name='zf', help='ROI zoom factor')
flags.DEFINE_integer('camera', 0, short_name='c', help='Camera ID (0,1,...)')
flags.DEFINE_string('file', None, short_name='f', help='Process video from given file')
flags.DEFINE_boolean('debug', False, help='Start with debug info')
flags.DEFINE_boolean('save_video', False, help='Start with video capture')
flags.DEFINE_boolean('save_roi', False, help='Save detected ROI images to out/roi directory')

HELP_INFO = 'Press ESC or Q to quit, S for camera settings, C for video capture, D for debug info'

def main(_):
    """ Video recognition pipeline """
    
    logger.setLevel(logging.INFO)
    show_welcome_screen()
    cv2.waitKey(10)

    class_names = fast_get_class_names()
    model = make_model(len(class_names))
    load_model(model)

    # This one is to warm up a model
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

        # Read 1 sec of cam video to let it initialize
        for _ in range(FPS_RATE):
            ret, _ = cam.read()
            if not ret:
                logger.error('Cannot read from camera, quitting')
                return

    ref_images = get_ref_images(class_names)

    frame_count = 0
    roi = None
    ref = None
    roi_label = None
    video_out = None
    show_debug = FLAGS.debug
    show_preprocessed = False
    logger.setLevel(logging.DEBUG if show_debug else logging.INFO)

    status_info = StatusInfo(max_len=2)
    if not FLAGS.file:
        status_info.append(HELP_INFO)

    if FLAGS.save_video:
        fn = os.path.join(OUTPUT_DIR, f'pipe_{datetime.now().strftime("%y%m%d_%H%M%S")}.mp4')
        status_info.append(f'Starting video output to {fn}')
        video_out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, tuple(reversed(FRAME_SIZE)))

    def detect_callback(roi: np.ndarray):
        preds = predict_image_probs(model, roi, class_names)
        if FLAGS.save_roi:
            save_roi(roi, preds[0][0], preds[0][1])
        return preds

    for detection in track_detect(
        cam, 
        detect_callback=detect_callback,
        track_time=3.0,
        replace_bg_color=(255,255,255)):

        if detection.label and detection.label != roi_label:
            # Detected piece has changed
            roi_label = detection.label
            roi = detection.roi
            ref = ref_images.get(detection.label)
            roi_caption = f'{detection.label} ({detection.prob:.2%})'
            status_info.append(f'Detection: {roi_caption}')
            if show_preprocessed:
                roi = preprocess_image(roi)
                ref = preprocess_image(ref)
            if show_debug:
                show_roi_window(roi, roi_caption)
                show_ref_window(ref, roi_label)

        frame = detection.frame
        if show_preprocessed:
            frame = preprocess_image(frame)
        else:
            status_info.apply(frame)
            if detection.bbox is not None:
                green_rect(frame, detection.bbox)

        show_frame(frame)

        if video_out is not None:
            video_out.write(frame)

        if frame_count % FPS_RATE == 0 and show_debug:
            show_hist_window(frame, roi, ref, log_scale=True)

        frame_count += 1
        key = int(cv2.waitKey(1) & 0xFF)
        match key:
            case 27 | 113:  # esc or q
                break

            case 32:    # space
                status_info.append('Paused, press SPACE to resume', True)
                status_info.apply(frame)
                show_frame(frame)
                while int(cv2.waitKey(10) & 0xFF) not in (27, 32, 113):
                    pass
                status_info.append('Resumed')

            case 99:    # c
                if video_out is None:
                    fn = os.path.join(OUTPUT_DIR, f'pipe_{datetime.now().strftime("%y%m%d_%H%M%S")}.mp4')
                    status_info.append(f'Starting video output to {fn}')
                    video_out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, tuple(reversed(FRAME_SIZE)))
                else:
                    video_out.release()
                    video_out = None
                    status_info.append('Video output stopped')

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
