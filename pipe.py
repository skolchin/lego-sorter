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
    cap = cv2.VideoCapture(-1)
    if not cap.isOpened():
        logger.error('Cannot open camera, exiting')
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == '__main__':
    app.run(main)
