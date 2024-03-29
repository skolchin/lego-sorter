# LEGO sorter project
# Video recognition pipeline demo
# (c) lego-sorter team, 2022-2023

import os
import cv2
import logging
from absl import app, flags
from datetime import datetime

from lib.globals import OUTPUT_DIR
from lib.pipe_utils import *
from lib.status_info import StatusInfo
from lib.object_tracker import track_detect, TrackObject, ObjectState, FrameCallbackReturn
from lib.model import load_model, make_model
from lib.image_dataset import fast_get_class_names, predict_image, predict_image_probs
from lib.controller import Controller
from lib.dummy_controller import DummyController

FLAGS = flags.FLAGS
flags.declare_key_flag('gray')
flags.declare_key_flag('edges')
flags.declare_key_flag('zoom')

del FLAGS.zoom_factor
flags.DEFINE_float('zoom_factor', 2.2, short_name='zf', help='ROI zoom factor')
flags.DEFINE_integer('camera', 0, short_name='c', help='Camera ID (0,1,...)')
flags.DEFINE_string('file', None, short_name='f', help='Process video from given file')
flags.DEFINE_boolean('debug', False, help='Start with debug info')
flags.DEFINE_boolean('save_video', False, help='Start with video capture')
flags.DEFINE_boolean('save_roi', False, help='Save all detected ROI images to out/roi directory')
flags.DEFINE_float('valid_confidence', 0.3, help='Confidence level to consider detection valid')
flags.DEFINE_boolean('dummy', False, help='Do not connect to HW controller')

HELP_INFO = 'Press ESC or Q to quit, S for camera settings, C for video capture, W to reclassify'

def main(_):
    """ Video recognition pipeline """

    # Show the welcome screen    
    logger.setLevel(logging.INFO)
    show_welcome_screen()
    cv2.waitKey(10)
    logger.setLevel(logging.DEBUG if FLAGS.debug else logging.INFO)

    # Load and warm up the model
    class_names = fast_get_class_names()
    model = make_model(len(class_names))
    load_model(model)

    frame = np.full(list(FRAME_SIZE) + [3], imu.COLOR_BLACK, np.uint8)
    predict_image(model, frame, class_names)

    # Initiazlize camera (either using pre-recorded file or connecting to real video cam)
    if FLAGS.file:
        logger.info(f'Processing video file {FLAGS.file}')
        cam = cv2.VideoCapture(FLAGS.file)
        controller = DummyController()
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
        apply_cam_props(cam)

        # Read 1 sec of cam video to let it initialize
        for _ in range(FPS_RATE):
            ret, _ = cam.read()
            if not ret:
                logger.error('Cannot read from camera, quitting')
                return

        # Initialize controller
        if FLAGS.dummy:
            controller = DummyController()
        else:
            controller = Controller()

    # Load reference images to display in debug mode
    ref_images = get_ref_images(class_names)

    # Assign class labels to bins (letters from A to Z)
    bins_map = {label: chr(ord('A')+n) for n, label in enumerate(class_names)}

    # Other init stuff
    frame_count = 0
    roi = None
    ref = None
    roi_label = None
    roi_prob = None
    video_out = None
    show_debug = FLAGS.debug
    show_preprocessed = False
    reset_tracker = False

    status_info = StatusInfo(max_len=2)
    if not FLAGS.file:
        status_info.append(HELP_INFO)

    if FLAGS.save_video:
        fn = os.path.join(OUTPUT_DIR, f'pipe_{datetime.now().strftime("%y%m%d_%H%M%S")}.mp4')
        status_info.append(f'Starting video output to {fn}')
        video_out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, tuple(reversed(FRAME_SIZE)))

    # Frame callback func, called on every frame
    def frame_callback(track_object: TrackObject) -> FrameCallbackReturn:
        nonlocal reset_tracker, roi_label
        if reset_tracker:
            reset_tracker = False
            return FrameCallbackReturn.RESET

        if track_object.state == ObjectState.NEW:
            roi_label = None
            controller.recognize()
        
        return FrameCallbackReturn.CONTINUE

    # Detection callback func, called to detect object on given frame
    def detect_callback(roi: np.ndarray):
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        preds = predict_image_probs(model, roi_rgb, class_names)
        label, prob = preds[0]
        if prob < FLAGS.valid_confidence:
            return None
        if FLAGS.save_roi:
            save_roi_out(roi, label, prob)
        return preds

    # Start conveiour
    controller.start_processing()

    # Main loop
    for detection in track_detect(
        cam, 
        frame_callback=frame_callback,
        detect_callback=detect_callback,
        track_time=1.0,
        replace_bg_color=BACK_COLOR):

        if detection.label and detection.label != roi_label:
            # Got new detection, proceed
            roi_label = detection.label
            roi = detection.roi
            roi_prob = detection.prob
            ref = ref_images.get(detection.label)
            roi_caption = f'{detection.label} ({detection.prob:.2%})'
            status_info.append(f'Detection: {roi_caption}', important=True)
            if show_preprocessed:
                roi = preprocess_image(roi)
                ref = preprocess_image(ref)
            if show_debug:
                show_roi_window(roi, roi_caption)
                show_ref_window(ref, roi_label)

            # Drop object to bin
            controller.select(bins_map[detection.label])

        # Show captured frame on screen, and, optionally, save to file
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

        # Debug info
        if frame_count % FPS_RATE == 0 and show_debug:
            show_hist_window(frame, roi, ref, log_scale=True)

        # Gather and process user input
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

            case 114:   # r
                reset_tracker = True
                status_info.append('Tracker reset')

            case 115:   # s
                if not FLAGS.file:
                    cam.set(cv2.CAP_PROP_SETTINGS, 1)

            case 119:   # w
                if roi is None or roi_label is None:
                    status_info.append('Cannot start label selection, detect something first!', True)
                else:
                    status_info.append('Waiting for label selection', True)
                    status_info.apply(frame)
                    show_frame(frame)
                    controller.wait()
                    cv2.waitKey(10)

                    new_label = choose_label(roi_label)
                    if not new_label:
                        status_info.append('No label selected')
                    else:
                        status_info.append(f'{new_label} label selected, saving for retraining')
                        save_roi_retrain(roi, new_label, roi_prob, roi_label)
                    controller.move()

    # Clean up
    if video_out is not None:
        video_out.release()
    cam.release()
    controller.stop_processing()

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s')
    app.run(main)
