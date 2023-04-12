# LEGO sorter project
# Object tracking functions
# (c) kol, 2022-2023

import os
import cv2
import logging
import numpy as np
from absl import flags
from enum import IntEnum
from time import time
from typing import List, Callable, Tuple

from .pipe_utils import bgmask_to_bbox, extract_roi

FLAGS = flags.FLAGS
flags.DEFINE_integer('var_threshold', 40, help='Background detection threshold')
flags.DEFINE_float('min_confidence', 0.1, help='Minimum confidence level to process the detection')
flags.DEFINE_float('valid_confidence', 0.3, help='Confidence level to consider detection valid')
flags.DEFINE_boolean('save_roi', False, help='Save ROIs detected to out/roi directory')

logger = logging.getLogger('lego-tracker')

def _init_back_sub(frame: np.ndarray):
    """ Internal - make a background subtractor instance and initialize it on a frame """
    back_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=FLAGS.var_threshold, detectShadows=True)
    back_sub.apply(frame, learningRate=-1)
    return back_sub

class ObjectState(IntEnum):
    """ States of object being detected """

    INITIAL = 0
    """ Nothing found """

    NEW = 1
    """ New object detected """

    MOVE = 2
    """ Object has moved from previous position """

    STOP = 3
    """ Object has stopped """

def track(cam: cv2.VideoCapture, replace_bg_color: Tuple[int] = None):
    """ Detect objects on video stream.

    This function continously monitors given video stream and detects objects coming in to the vision field. 
    It tracks an object detected while it is moving across the field and detects when it stops.
    The function yields every frame taken from the device along with detected object's bounding box 
    and state (new -> moved -> stopped). Only one object at a time could be tracked.
    
    Arguments:
        cam:    frame source (camera or video)
        replace_bg_color: if not `None`, background of stopped object will be erased and replaced with given color.
            Initially detected and moved objects will remain intact to avoid performace impact.

    Yields:
        `(frame, obj_bbox, obj_state)` tuple containing:
            `frame`: a frame obtained from the source,
            `obj_bbox`: bounding box of object detected or `None` if no objects found,
            `obj_state`: one of `ObjectState` constants.

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
                new_bgmask = obj_tracker.apply(frame, learningRate=-1)
                new_bbox = bgmask_to_bbox(new_bgmask)
                if new_bbox is None:
                    if obj_state == ObjectState.MOVE:
                        logger.info(f'Object stopped at {obj_bbox}')
                        obj_state = ObjectState.STOP
                else:
                    obj_bbox = new_bbox
                    obj_state = ObjectState.MOVE

            if replace_bg_color:
                frame[bgmask!=255] = replace_bg_color
                # frame = bgmask

        yield frame, obj_bbox, obj_state

def track_detect(
    cam: cv2.VideoCapture, 
    detect_callback: Callable[[np.ndarray], List],
    track_time: float = 2.0,
    replace_bg_color: Tuple[int] = None):
    """ Tracks and classifies objects on a video stream.

    This function continously monitors given frame source, detects objects coming in to the vision field
    and classifies them using given callback function. 

    When a new object is detected, the function collects ROIs of that object for some time and when the time elapses, 
    it uses a callback to classify the object on that ROI. After that, the detection loop is stopped until new object appears.

    The function yields every frame along with classification results. Only one object at a time could be tracked.
    
    Arguments:
        cam:    frame source (camera or video)
        detect_callback:    A callback which must accept a ROI of object detected and return
            a list contaiting `(label, probability)` tuples sorted descending on probability.
            Only 1st element of that iterable is used.
        track_time:     time limit to perform initial frame collection (in seconds)
        min_prob:   probability threshold

    Yields:
        `(frame, obj_bbox, detection)` tuple containing:
            `frame`: a frame obtained from the source,
            `obj_bbox`: bounding box of object detected or `None` if no objects found
            `detection`: a tuple containing ROI, label and probability.
        
    """

    detections = []
    detect_start_time = 0.0
    detection, detection_bbox = None, None
    frame_count = 0

    for (frame, bbox, state) in track(cam, replace_bg_color):
        frame_count += 1
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
                    if roi_prob > FLAGS.min_confidence:
                        detections.append((roi, roi_label, roi_prob, bbox))
                        if FLAGS.save_roi:
                            cv2.imwrite(os.path.join('out', 'roi', f'{roi_label}_{frame_count:04d}.png'), roi)

                elif detections:
                    probs = np.array([d[2] for d in detections])
                    idx = np.argmax(probs)
                    detection, detection_bbox = detections[idx][:-1], detections[idx][-1]
                    logger.debug(f'Detection loop finished, {detection[1]} label selected')
                    detections = []

                yield frame, detection_bbox, detection

            case _:
                yield frame, bbox, None
