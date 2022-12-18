# LEGO sorter project
# Recognition pipeline
# (c) kol, 2022

import cv2
import logging
import img_utils22 as imu
from absl import app, flags
from time import sleep

logging.basicConfig(format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import lib.globals
from lib.pipe_utils import *
FLAGS = flags.FLAGS
flags.declare_key_flag('gray')
flags.declare_key_flag('edges')

def main(argv):
    """ Video recognition pipeline """
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        logger.error('Cannot open camera, exiting')
        return

    cam.set(cv2.CAP_PROP_SETTINGS, 1)
    cam.set(cv2.CAP_PROP_FPS, 30.0)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow('frame', frame)
        if cv2.waitKey(30) & 0xFF in (27, ord('q')):
            break

    cam.release()

if __name__ == '__main__':
    app.run(main)
