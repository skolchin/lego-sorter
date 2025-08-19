# LEGO sorter project
# Camera object
# (c) lego-sorter team, 2022-2025

import cv2
import gevent
import logging
import numpy as np
from threading import Event, Lock, Thread

from lib.controller import Controller
from lib.object_tracker import TrackObject, ObjectState, track_detect
from lib.pipe_utils import FRAME_SIZE, FPS_RATE, green_rect

_logger = logging.getLogger(__name__)


class Camera:
    """ Camera encapsulation class """

    def __init__(self, controller: Controller, camera_id: int = 0) -> None:
        self.controller = controller
        self.cam = None
        self.video_thread = None

        self.stopCameraEvent = Event()
        self.frameReadyEvent = Event()
        self.captureBackgroundEvent = Event()

        self.lock = Lock()
        self.output_frame = None
        self.video_thread = None

        self.reset_camera(camera_id)

    def reset_camera(self, camera_id, auto_exposure=0, exposure=-10.0):
        self.camera_id = camera_id
        self.cam = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cam.isOpened():
            _logger.error('Cannot open camera, exiting')
            return

        self.cam.set(cv2.CAP_PROP_FPS, FPS_RATE)
        self.cam.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
        self.cam.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[1])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[0])
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def __gen_frames(self):
        """ Video thread main function """

        def frame_callback(track_object: TrackObject):
            if self.stopCameraEvent.is_set():
                # Stop event
                return False

            if track_object.state == ObjectState.NEW:
                self.controller.recognize()

            return True

        def detect_callback(frame: np.ndarray):
            self.controller.select('D')
            return []

        _logger.debug("Starting video stream")
        for detection in track_detect(
                self.cam,
                detect_callback=detect_callback,
                frame_callback=frame_callback):

            if detection.bbox is not None:
                green_rect(detection.frame, detection.bbox)

            if self.captureBackgroundEvent.is_set():
                self.captureBackgroundEvent.clear()

            with self.lock:
                self.output_frame = detection.frame.copy()

        _logger.debug("Video stream stopped")

    def start_video_stream(self):
        """ Start a video stream """

        self.video_thread = Thread(target=self.__gen_frames, name='video')
        self.video_thread.daemon = True
        self.stopCameraEvent.clear()
        self.video_thread.start()

        while True:
            # make gevent scheduler start another tasks
            gevent.sleep(0)

            # listen to stop camera event
            if self.stopCameraEvent.is_set():
                _logger.debug("CameraStop event received by get_video_stream")
                break

            # wait until the lock is acquired
            with self.lock:
                # check if the output frame is available, otherwise skip
                # the iteration of the loop
                if self.output_frame is None:
                    continue

                # encode the frame in JPEG format
                (flag, encodedImage) = cv2.imencode(".jpg", self.output_frame)
                # ensure the frame was successfully encoded
                if not flag:
                    continue

            # yield the output frame in the byte format
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    def capture_background(self):
        self.captureBackgroundEvent.set()

    def stop_video_stream(self):
        if self.video_thread is not None and self.video_thread.is_alive():
            self.stopCameraEvent.set()
            self.video_thread.join()
        self.video_thread = None

    @staticmethod
    def get_camera_indexes():
        # checks the first 10 indexes.
        index = 0
        arr = []
        i = 10
        while i > 0:
            # make gevent scheduler start another tasks
            gevent.sleep(0)
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                arr.append(index)
                cap.release()
            index += 1
            i -= 1
        return arr
