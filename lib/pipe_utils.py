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

SCREEN_SIZE = (480, 640)
SCREEN_TITLE = 'Stream'

def show_welcome_screen():
    screen = np.full(list(SCREEN_SIZE) + [3], imu.COLOR_BLACK, np.uint8)
    x,y = int(SCREEN_SIZE[1]/2) - 50, int(SCREEN_SIZE[0]/2) - 20
    cv2.putText(screen, 'Loading...', (x,y), cv2.FONT_HERSHEY_SIMPLEX, .8, color=imu.COLOR_GREEN)
    cv2.imshow(SCREEN_TITLE, screen)

def show_stream(img: np.ndarray):
    cv2.imshow(SCREEN_TITLE, img)

def hide_stream():
    cv2.destroyWindow(SCREEN_TITLE)

def show_status(img: np.ndarray, status: str, line: int = 0, important: bool = False):
    x, y = 10, 12 + 16 * line
    cv2.putText(img, status, (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, .6, color=imu.COLOR_RED if important else imu.COLOR_GREEN)

def green_rect(img: np.ndarray, bbox: tuple):
    x, y = bbox[0], bbox[1]
    cx, cy = bbox[0] + bbox[2], bbox[1] + bbox[3]
    cv2.rectangle(img, (x,y), (cx, cy), imu.COLOR_GREEN, 1)

def green_named_rect(img: np.ndarray, caption: str, bbox: tuple):
    x, y = bbox[0], bbox[1]
    cx, cy = bbox[0] + bbox[2], bbox[1] + bbox[3]
    tx, ty = x, y - 8
    cv2.putText(img, caption, (tx, ty), cv2.FONT_HERSHEY_COMPLEX_SMALL, .6, color=imu.COLOR_GREEN)
    cv2.rectangle(img, (x,y), (cx, cy), imu.COLOR_GREEN, 1)

def bgmask_to_bbox(bg_mask: np.ndarray) -> tuple:
    kernel = imu.misc.get_kernel(10)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bg_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):
        return None

    areas = [cv2.contourArea(c) for c in contours]
    contour = contours[np.argmax(areas)]
    return cv2.boundingRect(contour)

def get_bbox_area(img: np.ndarray, bbox: tuple, bbox_relax) -> np.ndarray:
    if isinstance(bbox_relax, float):
        dw, dh = int(bbox[2]*bbox_relax), int(bbox[3]*bbox_relax)
    else:
        dw, dh = bbox_relax, bbox_relax

    bbox = [
        max(bbox[0] - dw, 0),
        max(bbox[1] - dh, 0),
        min(bbox[0] + bbox[2] + dw, img.shape[1]),
        min(bbox[1] + bbox[3] + dh, img.shape[0])
    ]
    return imu.get_image_area(img, bbox)
    