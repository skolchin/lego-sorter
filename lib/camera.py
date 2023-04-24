import logging
from threading import Event, Lock, Thread
import cv2
from lib.controller import Controller
from lib.pipe_utils import *
import gevent


class Camera():
    logger = logging.getLogger(__name__)

    CAMERA_ID = 0
    cam = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    stopCameraEvent = Event()
    frameReadyEvent = Event()
    captureBackgroundEvent = Event()

    lock = Lock()
    output_frame = None
    video_thread = None

    def __init__(self) -> None:
        self.reset_camera(self.CAMERA_ID)

    def reset_camera(self, cam_id, auto_exposure=0, exposure=-10.0):
        if not self.cam.isOpened():
            logger.error('Cannot open camera, exiting')
            return

        self.cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

        self.cam.set(cv2.CAP_PROP_FPS, FPS_RATE)
        self.cam.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
        self.cam.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[1])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[0])
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure)

        self.stop()

        self.video_thread = Thread(target=self.__gen_frames, name='video')
        self.video_thread.daemon = True
        self.stopCameraEvent.clear()
        logger.info("Starting video thread")
        self.video_thread.start()
        logger.info("Video thread started")

    def init_back_sub(self, frame):
        back_sub = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=20.0, detectShadows=True)
        back_sub.apply(frame, learningRate=1)
        return back_sub

    def __gen_frames(self):
        frame_count = 0
        back_sub = None
        obj_tracker = None
        obj_bbox = None
        obj_moved = False

        controller = Controller()

        while True:
            if self.stopCameraEvent.is_set():
                logger.info("CameraStop event recieved by gen_frames")
                break

            ret, frame = self.cam.read()
            if not ret:
                self.logger.error('Cannot grab frame from camera, exiting')
                break

            if back_sub is not None:
                # Detect any changes to static background
                bgmask = back_sub.apply(frame, learningRate=0)
                obj_bbox = bgmask_to_bbox(bgmask)
                if obj_bbox is None:
                    # Nothing found
                    if obj_tracker is not None:
                        # Object is gone
                        self.logger.info('Object has left the building')
                        obj_tracker = None

                elif obj_bbox[2] == FRAME_SIZE[1] and obj_bbox[3] == FRAME_SIZE[0]:
                    self.logger.error('Object is too big, resetting the pipe')
                    back_sub = self.init_back_sub(frame)
                    obj_bbox = None
                    obj_tracker = None

                else:
                    # Got something
                    if obj_tracker is None:
                        # New object, setup a tracker
                        self.logger.info(f'New object detected at {obj_bbox}')

                        controller.recognize()

                        obj_tracker = self.init_back_sub(frame)
                        obj_moved = True
                    else:
                        # Object has already been tracked, check it has not been moved from last position
                        bgmask = obj_tracker.apply(frame, learningRate=-1)
                        new_bbox = bgmask_to_bbox(bgmask)
                        if new_bbox is None:
                            if obj_moved:
                                self.logger.info(
                                    f'Object stopped at {obj_bbox}')
                                obj_moved = False

                                # Detect label and proceed with controller
                                # label = choice(self.controller.labels)
                                # TODO: remove and replace with label
                                controller.select("D")
                        else:
                            obj_bbox = new_bbox
                            obj_moved = True

            if obj_bbox is not None:
                green_rect(frame, obj_bbox)

            frame_count += 1

            if self.captureBackgroundEvent.is_set():
                self.captureBackgroundEvent.clear()

                back_sub = self.init_back_sub(frame)
                obj_bbox = None
                obj_tracker = None
                self.logger.info('Background captured')

            with self.lock:
                self.output_frame = frame.copy()

    def get_video_stream(self):
        while True:
            # make gevent scheduler start another tasks
            gevent.sleep(0)

            # listen to stop camera event
            if self.stopCameraEvent.is_set():
                logger.info("CameraStop event received by get_video_stream")
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

    def stop(self):
        if self.video_thread is not None and self.video_thread.is_alive():
            self.stopCameraEvent.set()
            self.video_thread.join()

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
