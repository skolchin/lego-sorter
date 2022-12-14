# LEGO sorter project
# Pipeline support utils
# (c) kol, 2022

import os
import cv2
import numpy as np
import img_utils22 as imu
import logging
from functools import cache
from typing import Iterable, Tuple

from .globals import IMAGE_DIR
from .image_dataset import _preprocess
from .status_info import StatusInfo

logger = logging.getLogger(__name__)

FPS_RATE = 30
FRAME_SIZE = (480, 640)
FRAME_WINDOW_TITLE = 'Frame'
ROI_WINDOW_TITLE = 'ROI'
ROI_WINDOW_SIZE = (240, 320)
HIST_WINDOW_SIZE = (240, 320)
HIST_WINDOW_TITLE = 'Histogram'
REF_WINDOW_SIZE = (240, 320)
REF_WINDOW_TITLE = 'Ref'

def show_welcome_screen():
    frame = np.full(list(FRAME_SIZE) + [3], imu.COLOR_BLACK, np.uint8)
    x,y = int(FRAME_SIZE[1]/2) - 50, int(FRAME_SIZE[0]/2) - 20
    cv2.putText(frame, 'Loading...', (x,y), cv2.FONT_HERSHEY_SIMPLEX, .8, color=imu.COLOR_GREEN)
    cv2.imshow(FRAME_WINDOW_TITLE, frame)

def show_frame(frame: np.ndarray):
    cv2.imshow(FRAME_WINDOW_TITLE, frame)

def hide_stream():
    cv2.destroyWindow(FRAME_WINDOW_TITLE)

def show_roi_window(img: np.ndarray, caption: str, contour: Iterable = None, wsize: Tuple[int] = ROI_WINDOW_SIZE):
    if img is None:
        canvas = np.full(list(wsize) + [3], imu.COLOR_GRAY, np.uint8)
    else:
        canvas = np.full(list(wsize) + [3], imu.COLOR_WHITE, np.uint8)
        if img.shape[0] > canvas.shape[0] or img.shape[1] > canvas.shape[1]:
            scale_y, scale_x = (canvas.shape[0]-20) / img.shape[0], (canvas.shape[1]-20) / img.shape[1]
            img = imu.resize(img, scale=min(scale_x, scale_y))
        y = max(int((canvas.shape[0]-20)/2 - img.shape[0]/2), 0)
        x = max(int(canvas.shape[1]/2 - img.shape[1]/2), 0)
        try:
            if len(img.shape) == 2: 
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            canvas = imu.patch(canvas, x, y, img, clip=True)
        except Exception as ex:
            logger.exception('Error %s', ex, exc_info=1)
        StatusInfo().assign_and_apply(canvas, caption, important=True)

    cv2.imshow(ROI_WINDOW_TITLE, canvas)

def hide_roi_window():
    try:
        cv2.destroyWindow(ROI_WINDOW_TITLE)
    except:
        pass

def show_hist_window(frame: np.ndarray, roi: np.ndarray = None, ref: np.ndarray = None, 
    wsize: Tuple[int] = HIST_WINDOW_SIZE, log_scale: bool=False):

    img_list = [frame]
    if roi is not None: img_list += [roi]
    if ref is not None: img_list += [ref]

    hist = plot_hist(img_list, wsize, log_scale)
    cv2.imshow(HIST_WINDOW_TITLE, hist)

def hide_hist_window():
    try:
        cv2.destroyWindow(HIST_WINDOW_TITLE)
    except:
        pass

def show_ref_window(ref: np.ndarray, caption: str, wsize: Tuple[int] = REF_WINDOW_SIZE):
    if ref is None:
        hide_ref_window()
    else:
        ref = imu.resize(ref, wsize)
        cv2.imshow(REF_WINDOW_TITLE, ref)

def hide_ref_window():
    try:
        cv2.destroyWindow(REF_WINDOW_TITLE)
    except:
        pass

def green_rect(frame: np.ndarray, bbox: Tuple[int]):
    x, y = bbox[0], bbox[1]
    cx, cy = bbox[0] + bbox[2], bbox[1] + bbox[3]
    cv2.rectangle(frame, (x,y), (cx, cy), imu.COLOR_GREEN, 1)

def green_named_rect(frame: np.ndarray, bbox: Tuple[int], caption: str) -> np.ndarray:
    x, y = bbox[0], bbox[1]
    cx, cy = bbox[0] + bbox[2], bbox[1] + bbox[3]
    tx, ty = x, y - 8
    frame = cv2.putText(frame, caption, (tx, ty), cv2.FONT_HERSHEY_COMPLEX_SMALL, .6, color=imu.COLOR_GREEN)
    frame = cv2.rectangle(frame, (x,y), (cx, cy), imu.COLOR_GREEN, 1)
    return frame

def bgmask_to_bbox_contour(bg_mask: np.ndarray) -> Tuple[Tuple[int], Iterable]:
    kernel = imu.misc.get_kernel(10)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bg_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):
        return None, None

    areas = [cv2.contourArea(c) for c in contours]
    contour = contours[np.argmax(areas)]
    bbox = cv2.boundingRect(contour)
    logger.debug(f'{len(contours)} contours detected, max bbox is {bbox}')

    return bbox, contour

def bgmask_to_bbox(bg_mask: np.ndarray) -> Tuple[int]:
    return bgmask_to_bbox_contour(bg_mask)[0]

def extract_roi(frame: np.ndarray, bbox: Tuple[int], bbox_relax=0.2, zoom_level=2.0) -> np.ndarray:
    dw, dh = int(bbox[2]*bbox_relax), int(bbox[3]*bbox_relax)
    bbox = [
        max(bbox[0] - dw, 0),
        max(bbox[1] - dh, 0),
        min(bbox[0] + bbox[2] + dw, frame.shape[1]),
        min(bbox[1] + bbox[3] + dh, frame.shape[0])
    ]
    logger.debug(f'Normalized bbox: {bbox}')
    roi = imu.get_image_area(frame, bbox)
    if zoom_level > 0.0:
        roi = imu.rescale(roi, scale=zoom_level, center=True, pad_color=imu.COLOR_WHITE)
    return roi

def plot_hist(img_list: list, wsize: Tuple[int], log_scale: bool = False) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.cm import get_cmap

    fig = Figure()
    cmap = get_cmap('Paired')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim([0,256])
    ax.margins(0)
    if log_scale: 
        ax.set_yscale('log')

    for img, clr in zip(img_list, cmap.colors):
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        ax.plot(hist, color=clr)

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    return imu.resize(buf, wsize)

def preprocess_image(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return None
    frame = _preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))[0]
    if hasattr(frame, 'numpy'): frame = frame.numpy()
    return frame.astype('uint8')

@cache
def get_ref_image(label: str) -> np.ndarray:
    fn = os.path.join(IMAGE_DIR, label, f'{label}_0.png')
    try:
        return cv2.imread(fn)
    except FileNotFoundError:
        logger.error(f'Cannot find file {fn}')
        return None
