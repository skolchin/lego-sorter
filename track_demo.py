# LEGO sorter project
# Object tracking demo
# (c) lego-sorter team, 2022-2023

import cv2
import logging
from absl import app, flags

from lib.pipe_utils import show_frame, green_rect, FPS_RATE, FRAME_SIZE
from lib.object_tracker import track

FLAGS = flags.FLAGS
flags.DEFINE_string('file', None, short_name='f', help='Process video from given file')
flags.DEFINE_integer('camera', 0, short_name='c', help='Camera ID')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(_):
    if FLAGS.file:
        cam = cv2.VideoCapture(FLAGS.file)
    else:
        cam = cv2.VideoCapture(FLAGS.camera, cv2.CAP_DSHOW)
        if not cam.isOpened():
            logger.error('Cannot open camera, exiting')
            return

        cam.set(cv2.CAP_PROP_FPS, FPS_RATE)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[1])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[0])

    frame_count = 0
    for tro in track(cam, replace_bg_color=(255,255,255)):
        if tro.bbox is not None:
            green_rect(tro.frame, tro.bbox)

        show_frame(tro.frame)
        frame_count += 1

        key = int(cv2.waitKey(1) & 0xFF)
        match key:
            case 27 | 113:  # esc or q
                break

            case 115:   # s
                if not FLAGS.file:
                    cam.set(cv2.CAP_PROP_SETTINGS, 1)


if __name__ == '__main__':
    app.run(main)
