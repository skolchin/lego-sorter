# LEGO sorter project
# Object tracking demo
# (c) kol, 2022

import cv2
import logging
from random import choice
from absl import app, flags

from lib.pipe_utils import *
from lib.controller import Controller

logger = logging.getLogger(__name__)

CAMERA_ID = 0

def init_back_sub(frame):
    back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20.0, detectShadows=True)
    back_sub.apply(frame, learningRate=1)
    return back_sub

def main(_):
    logger.setLevel(logging.INFO)
    show_welcome_screen()
    cv2.waitKey(10)

    cam = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    if not cam.isOpened():
        logger.error('Cannot open camera, exiting')
        return

    cam.set(cv2.CAP_PROP_FPS, FPS_RATE)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[1])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[0])

    frame_count = 0
    back_sub = None
    obj_tracker = None
    obj_bbox = None
    obj_moved = False

    controller = Controller()

    while True:
        ret, frame = cam.read()
        if not ret:
            logger.error('Cannot grab frame from camera, exiting')
            break

        if back_sub is not None:
            # Detect any changes to static background
            bgmask = back_sub.apply(frame, learningRate=0)
            obj_bbox = bgmask_to_bbox(bgmask)
            if obj_bbox is None:
                # Nothing found
                if obj_tracker is not None:
                    # Object is gone
                    logger.info('Object has left the building')
                    obj_tracker = None
                    controller.start()

            elif obj_bbox[2] == FRAME_SIZE[1] and obj_bbox[3] == FRAME_SIZE[0]:
                logger.error('Object is too big, resetting the pipe')
                back_sub = init_back_sub(frame)
                obj_bbox = None
                obj_tracker = None

            else:
                # Got something
                if obj_tracker is None:
                    # New object, setup a tracker
                    logger.info(f'New object detected at {obj_bbox}')
                    controller.stop()
                    obj_tracker = init_back_sub(frame)
                    obj_moved = True
                else:
                    # Object has already been tracked, check it has not been moved from last position
                    bgmask = obj_tracker.apply(frame, learningRate=-1)
                    new_bbox = bgmask_to_bbox(bgmask)
                    if new_bbox is None:
                        if obj_moved:
                            logger.info(f'Object stopped at {obj_bbox}')
                            obj_moved = False

                            # Detect label and proceed with controller
                            label = choice(controller.labels)
                            controller.process_object(label)
                    else:
                        obj_bbox = new_bbox
                        obj_moved = True

        if obj_bbox is not None:
            green_rect(frame, obj_bbox)

        show_frame(frame)
        frame_count += 1

        key = int(cv2.waitKey(1) & 0xFF)
        match key:
            case 27 | 113:  # esc or q
                break

            case 98:    # b
                back_sub = init_back_sub(frame)
                obj_bbox = None
                obj_tracker = None
                logger.info('Background captured')
                controller.start()

            case 115:   # s
                cam.set(cv2.CAP_PROP_SETTINGS, 1)

if __name__ == '__main__':
    app.run(main)
