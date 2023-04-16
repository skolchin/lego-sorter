import logging
import cv2
from lib.pipe_utils import *
from lib.exceptions import NotImplemented
import eventlet


class Camera():

    logger = logging.getLogger(__name__)
    CAMERA_ID = 0

    cam = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    need_capture_back_flag = False

    stop_callback = NotImplemented.raise_this
    process_callback = NotImplemented.raise_this

    frame_buffer = []
    frame_ready_semaphore = False
    camera_active = False

    def __init__(self, process_callback, stop_callback) -> None:
        if not self.cam.isOpened():
            logger.error('Cannot open camera, exiting')
            return

        self.cam.set(cv2.CAP_PROP_FPS, FPS_RATE)
        self.cam.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
        self.cam.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[1])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[0])
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, -10.0)

        self.process_callback = process_callback
        self.stop_callback = stop_callback

    def init_back_sub(frame):
        back_sub = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=20.0, detectShadows=True)
        back_sub.apply(frame, learningRate=1)
        return back_sub

    def gen_frames(self):
        frame_count = 0
        back_sub = None
        obj_tracker = None
        obj_bbox = None
        obj_moved = False

        self.camera_active = True

        while self.camera_active:
            eventlet.sleep(0)
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
                        self.stop_callback()
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
                                self.process_callback()
                        else:
                            obj_bbox = new_bbox
                            obj_moved = True

            if obj_bbox is not None:
                green_rect(frame, obj_bbox)

            show_frame(frame)
            frame_count += 1

            if self.need_capture_back_flag:
                self.need_capture_back_flag = False

                back_sub = self.init_back_sub(frame)
                obj_bbox = None
                obj_tracker = None
                self.logger.info('Background captured')

            ret, buffer = cv2.imencode('.jpg', frame)
            self.frame_buffer = buffer.tobytes()
            self.frame_ready_semaphore = True
            # frame = buffer.tobytes()
            # yield (b'--frame\r\n'
            #       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    def get_video_stream(self):
        while self.camera_active:
            eventlet.sleep(0)
            if self.frame_ready_semaphore:
                self.frame_ready_semaphore = False

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self.frame_buffer + b'\r\n')  # concat frame one by one and show result

    def capture_background(self):
        self.need_capture_back_flag = True

    def stop(self):
        self.camera_active = False

    @staticmethod
    def get_camera_indexes():
        # checks the first 10 indexes.
        index = 0
        arr = []
        i = 10
        while i > 0:
            eventlet.sleep(0)
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                arr.append(index)
                cap.release()
            index += 1
            i -= 1
        return arr
