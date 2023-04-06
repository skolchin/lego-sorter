# LEGO sorter project
# Object tracking functions
# (c) kol, 2022-2023

import cv2
import logging
import numpy as np
from absl import flags
from enum import IntEnum
from time import time
from typing import Any, Iterable, Callable

from .pipe_utils import bgmask_to_bbox, extract_roi

FLAGS = flags.FLAGS
flags.DEFINE_integer('var_threshold', 15, help='Background detection threshold')

logger = logging.getLogger('lego-tracker')

def _init_back_sub(frame: np.ndarray):
    """ Internal - make a background subtractor instance and initialize it on a frame """
    back_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=FLAGS.var_threshold, detectShadows=True)
    back_sub.apply(frame, learningRate=-1)
    return back_sub

class ObjectState(IntEnum):
    """ States of object beeing detected """

    INITIAL = 0
    """ Nothing found """

    NEW = 1
    """ New object detected """

    MOVE = 2
    """ Object has moved from last position """

    STOP = 3
    """ Object has stopped """

def track(cam: cv2.VideoCapture):
    """ Detect objects on video stream.

    This function continously monitors given video stream and detects objects coming in to the vision field
    yielding every frame along with detected object's bounding box and state. 
    Only one object at a time could be tracked.
    
    Arguments:
        cam:    frame source (camera or video)

    Yields:
        `(frame, obj_bbox, obj_state)` tuple, where:
            `frame` is a frame obtained from the source,
            `obj_bbox` is bounding box of object detected or `None` if no objects found,
            `obj_state`, which is one of `ObjectState` constants.

    Examples:
        >>> cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        >>> for (frame, bbox, state) in track(cam):
        >>>     if bbox is not None:
        >>>         draw_bounding_box(frame, bbox)
        >>>     cv2.imshow('frame', frame)
        
    """
    back_sub = None
    obj_tracker = None
    obj_bbox = None
    obj_state: ObjectState = ObjectState.INITIAL

    while True:
        ret, frame = cam.read()
        if not ret:
            logger.error('No more frames, exiting')
            break

        # Make a static background subtractor
        if back_sub is None:
            back_sub = _init_back_sub(frame)

        # Detect any changes to static background
        bgmask = back_sub.apply(frame, learningRate=0)
        obj_bbox = bgmask_to_bbox(bgmask)
        if obj_bbox is None:
            # Nothing found
            if obj_tracker is not None:
                # Object is gone
                logger.info('Object has left the building')
                obj_tracker = None
                obj_state = ObjectState.INITIAL
        else:
            # Got something
            if obj_tracker is None:
                # New object, setup a tracker
                logger.info(f'New object detected at {obj_bbox}')
                obj_tracker = _init_back_sub(frame)
                obj_state = ObjectState.NEW
            else:
                # Object has already been tracked, check it has not been moved from last position
                bgmask = obj_tracker.apply(frame, learningRate=-1)
                new_bbox = bgmask_to_bbox(bgmask)
                if new_bbox is None:
                    if obj_state == ObjectState.MOVE:
                        logger.info(f'Object stopped at {obj_bbox}')
                        obj_state = ObjectState.STOP
                else:
                    obj_bbox = new_bbox
                    obj_state = ObjectState.MOVE

        yield frame, obj_bbox, obj_state

def track_detect(
    cam: cv2.VideoCapture, 
    detect_callback: Callable[[np.ndarray], Iterable],
    track_time: float = 2.0, 
    min_prob: float = 0.1):
    """ Tracks and classifies objects on a video stream.

    This function continously monitors given frame source, detects objects coming in to the vision field
    and classifies them using given callback function. The function yields every frame along with classification results.
    Only one object at a time could be tracked.

    When a new object is detected, the function collects ROIs of that object and calls provided callback
    to classify the object on that ROI. ROI collection and detection is performed for specified period of time (2 sec),
    after that detection is stopped until new object appears.
    
    Arguments:
        cam:    frame source (camera or video)
        detect_callback:    A callback function. The function accepts a ROI of object and must return
            an iterable contaiting (label, probability) tuples sorted descending on probability.
            Only 1st element of that iterable is used.
        track_time:     time limit to perform initial frame collection (seconds)
        min_prob:   probability threshold

    Yields:
        `(frame, obj_bbox, detection)` tuple, where:
            `frame` is a frame obtained from the source,
            `obj_bbox` is bounding box of object detected or `None` if no objects found and
            `detection` is a tuple containing ROI, label and probability.

        
    """

    detections = []
    detect_start_time = 0.0
    detection, detection_bbox = None, None

    for (frame, bbox, state) in track(cam):
        if not bbox:
            continue

        match state:
            case ObjectState.NEW:
                # New object detected
                detections = []
                detect_start_time = 0.0
                detection = None
                yield frame, bbox, None

            case ObjectState.STOP:
                # Object has stopped under the cam
                if not detect_start_time:
                    detect_start_time = time()
                    logger.debug('Detection loop started')

                if time() <= detect_start_time + track_time:
                    roi = extract_roi(frame, bbox, zoom=FLAGS.zoom_factor)
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    labels_probs = detect_callback(roi_rgb)
                    logger.debug(f'Top-3 detections: {labels_probs[:3]}')
                    roi_label, roi_prob = labels_probs[0]
                    if roi_prob > min_prob:
                        detections.append((roi, roi_label, roi_prob, bbox))

                elif detections:
                    probs = np.array([d[2] for d in detections])
                    idx = np.argmax(probs)
                    detection, detection_bbox = detections[idx][:-1], detections[idx][-1]
                    logger.debug(f'Detection loop finished, {detection[1]} label selected')
                    detections = []

                yield frame, detection_bbox, detection

            case _:
                yield frame, bbox, None
