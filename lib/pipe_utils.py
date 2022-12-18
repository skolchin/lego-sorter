# LEGO sorter project
# Pipeline support utils
# (c) kol, 2022

import cv2
import logging
import numpy as np
import img_utils22 as imu

logging.basicConfig(format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SCREEN_SIZE = (600, 600)
SCREEN_TITLE = 'Stream'

def show_welcome_screen(caption: str = None) -> str:
    if caption is None or '...' in caption:
        caption = 'Loading'
    else:
        caption += '.'
    
    screen = np.full(list(SCREEN_SIZE) + [3], imu.COLOR_BLACK, np.uint8)
    x,y = int(SCREEN_SIZE[1]/2) - 50, int(SCREEN_SIZE[0]/2) - 20
    cv2.putText(screen, caption, (x,y), cv2.FONT_HERSHEY_SIMPLEX, .8, color=imu.COLOR_GREEN)
    cv2.imshow(SCREEN_TITLE, screen)
    return caption

