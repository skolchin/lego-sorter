# LEGO sorter project
# Object tracking functions
# (c) lego-sorter team, 2022-2023

import cv2
import logging
import numpy as np
from absl import flags
from enum import IntEnum
from time import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Callable, Tuple

from .pipe_utils import bgmask_to_bbox, extract_roi

FLAGS = flags.FLAGS
flags.DEFINE_integer('var_threshold', 40, help='Background detection threshold')

_logger = logging.getLogger('lego-tracker')

class ObjectState(IntEnum):
    """ States of object being detected """

    NONE = 0
    """ Nothing found """

    NEW = 1
    """ New object detected """

    MOVE = 2
    """ Object has moved from previous position """

    STOP = 3
    """ Object has stopped """

@dataclass
class TrackObject:
    """ Object tracking results """

    frame: np.ndarray
    """ Actual frame """

    bgmask: Any = None
    """ Object mask (`None` if no objects found) """

    bbox: Tuple[int] = None
    """ bounding box of detected object or `None` if nothing found """

    state: ObjectState = ObjectState.NONE
    """ Detection state """

@dataclass
class Detection:
    """ Detection results """

    frame: np.ndarray
    """ Actual frame """

    bbox: Tuple[int] = None
    """ ROI bbox deduced from bgmask (`None` if no objects found) """

    roi: np.ndarray = None
    """ Region of interest (ROI) image (`None` if no objects found) """

    label: str = None
    """ Label detected (`None` if nothing detected) """

    prob: float = None
    """ Detection probability (`None` if nothing detected) """


def _init_back_sub(frame: np.ndarray):
    """ Internal - make a background subtractor instance and initialize it on a frame """
    back_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=FLAGS.var_threshold, detectShadows=True)
    back_sub.apply(frame, learningRate=-1)
    return back_sub

def _max_prob_detection(detections: List[Detection]) -> Detection:
    """ Internal - choose detection with maximum probability """
    probs = np.array([d.prob for d in detections])
    idx = np.argmax(probs)
    return detections[idx]

def _max_rating_detection(detections: List[Detection]) -> Detection:
    """ Internal - choose detection with maximum rating """
    counter, last_entry = defaultdict(list), {}
    for n, d in enumerate(detections):
        counter[d.label].append(d.prob)
        last_entry[d.label] = n

    rating = {k: sum(v) for k, v in counter.items()}
    _logger.debug(f'Rating: {rating}')

    label = max(rating, key=lambda x: rating[x])
    return detections[last_entry[label]]

def track(cam: cv2.VideoCapture, 
            replace_bg_color: Tuple[int] = None,
            frame_callback: Callable[[TrackObject], bool] = None):
    """ Detect and track an object in video stream.

    The function continously monitors given video stream and detects objects coming in to the vision field. 
    It tracks an object detected while it is moving across the vision field and detects when it stops.

    The function yields every frame taken from the device along with detected object's bounding box 
    and state (new -> moved -> stopped). Only one object at a time could be tracked.
    
    Arguments:
        cam:    frame source (camera or video)
        replace_bg_color: if not `None`, object background (everything except object itself) 
            will be erased and replaced with given color
        frame_callback: a callback function, which receives a `TrackObject` instance and
            should return either `True` to continue tracking or `False` to immediatelly
            stop the loop and exit

    Yields:
        `TrackObject` object

    Returns:
        `True` if no more frames from video stream were received, `False` if terminated with callback

    Examples:
        >>> cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        >>> for tro in track(cam):
        >>>     if bbox is not None:
        >>>         draw_bounding_box(tro.frame, tro.bbox)
        >>>     cv2.imshow('frame', tro.frame)
        
    """
    back_sub = None
    obj_tracker = None
    obj_bbox = None
    obj_state: ObjectState = ObjectState.NONE

    while True:
        ret, frame = cam.read()
        if not ret:
            _logger.info('No more frames, exiting')
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
                _logger.debug('Object has left the building')
                obj_tracker = None
                obj_state = ObjectState.NONE
        else:
            # Got something
            if obj_tracker is None:
                # New object, setup a tracker
                _logger.debug(f'New object detected at {obj_bbox}')
                obj_tracker = _init_back_sub(frame)
                obj_state = ObjectState.NEW
            else:
                # Object has already been tracked, check it has not been moved from last position
                new_bgmask = obj_tracker.apply(frame, learningRate=-1)
                new_bbox = bgmask_to_bbox(new_bgmask)
                if new_bbox is None:
                    if obj_state == ObjectState.MOVE:
                        _logger.debug(f'Object stopped at {obj_bbox}')
                        obj_state = ObjectState.STOP
                else:
                    obj_bbox = new_bbox
                    obj_state = ObjectState.MOVE

            if replace_bg_color:
                frame[bgmask!=255] = replace_bg_color

        tro = TrackObject(frame, bgmask, obj_bbox, obj_state)
        if frame_callback and not frame_callback(tro):
            return False
        
        yield tro

    return True

def track_detect(
    cam: cv2.VideoCapture, 
    detect_callback: Callable[[np.ndarray], List],
    track_time: float = 2.0,
    replace_bg_color: Tuple[int] = None,
    use_rating: bool = True,
    frame_callback: Callable[[TrackObject], bool] = None):
    """ Detects and classifies objects in a video stream.

    This function continously monitors given frame source, detects objects coming in to the vision field
    and classifies them using given callback function. 

    When a new object is detected, the function collects ROIs of that object for some time. 
    After the time elapses, it chooses the best classification match and stops the detection loop 
    until new object appears.

    The function yields every frame along with detection results. Only one object at a time could be tracked.

    Arguments:
        cam:    frame source (camera or video)
        detect_callback:    A callback which must accept a ROI of an object detected and return
            a list contaiting `(label, probability)` tuples sorted descending on probability.
            Only 1st element of that iterable is currently used.
        track_time:     time limit to perform initial frame collection (in seconds)
        replace_bg_color: if not `None`, background of an ROI will be erased and replaced with given color
            before passing it to the callback
        use_rating: if `True` (default), uses rating mechanism to determine best detection from collected list.
            Otherwise, selects detection with highest probability.
        frame_callback: a callback function, which receives a track detection object and
            should return either `True` to continue tracking or `False` to immediatelly
            stop the loop and exit

    Yields:
        `Detection` object
        
    """

    detections: List[Detection] = []
    detect_start_time = 0.0
    detection: Detection = None
    frame_count = 0

    for tro in track(cam, frame_callback=frame_callback):
        frame_count += 1
        detection = Detection(tro.frame, tro.bbox)
        
        match tro.state:
            case ObjectState.NEW:
                # New object detected
                detections = []
                detect_start_time = 0.0

            case ObjectState.STOP:
                # Object has stopped
                if not detect_start_time:
                    detect_start_time = time()
                    _logger.debug('Detection loop started')

                if time() <= detect_start_time + track_time:
                    # Time didn't elapse yet, collect predictions for further selection
                    # No detections are emitted while detection loop is running
                    if not replace_bg_color:
                        frame = tro.frame
                    else:
                        frame = tro.frame.copy()
                        frame[tro.bgmask!=255] = replace_bg_color

                    roi = extract_roi(frame, tro.bbox, zoom=FLAGS.zoom_factor)
                    labels_probs = detect_callback(roi)
                    if not labels_probs:
                        _logger.debug('Got no detections, ignoring object')
                    else:
                        _logger.debug(f'Top-3 detections: {labels_probs[:3]}')
                        roi_label, roi_prob = labels_probs[0]
                        detections.append(Detection(tro.frame, tro.bbox, roi, roi_label, roi_prob))

                elif detections:
                    # Time's out, find best prediction and pass it to the caller
                    detection = _max_rating_detection(detections) if use_rating else _max_prob_detection(detections)
                    _logger.debug(f'Detection loop finished, {detection.label} label selected')
                    detections = []

        yield detection
