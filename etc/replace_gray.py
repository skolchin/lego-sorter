# LEGO sorter project
# Replace gray color from images
# (c) lego-sorter team, 2022-2023

import cv2
import logging
import numpy as np
import img_utils22 as imu
from pathlib import Path

logging.basicConfig(format='%(levelname)s: %(message)s')
logger = logging.getLogger('lego-tracker')
logger.setLevel(logging.INFO)

def is_debug() -> bool:
    return logger.getEffectiveLevel() == logging.DEBUG

def debug_imshow(img: np.ndarray, title: str = None):
    if is_debug():
        imu.imshow(img, title, (600, 800))

def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)

def find_bgmask_multichannel(img: np.ndarray) -> np.ndarray:

    # Find most popular color, which is assumed to be background
    bgclr = bincount_app(img)
    logger.info('Background color is %s', bgclr)

    # Build a background subtraction mask
    bg_mask = imu.get_bgsub_mask(img, np.full(img.shape, bgclr, img.dtype))
    nonzero = np.nonzero(bg_mask)
    if not nonzero[0].any() or not nonzero[1].any():
        raise Exception('ERROR: only background color detected')
    debug_imshow(bg_mask, 'initial bg_mask')

    # Flood fill holes within the mask starting from all 4 corners
    # (there usually comes ligthing artifacts)
    bg_mask[ bg_mask != 0 ] = 255
    corners = [
        (0, 0), 
        (0, bg_mask.shape[0]-1), 
        (bg_mask.shape[1]-1, 0), 
        (bg_mask.shape[1]-1, bg_mask.shape[0]-1)
    ]
    for pt in corners:
        cv2.floodFill(bg_mask, None, pt, 0)
    debug_imshow(bg_mask, 'filled bg_mask')
    return bg_mask

def find_bgmask_luminosity(img: np.ndarray, process_channel=2) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channels = cv2.split(lab)
    for n, c in enumerate(channels):
        debug_imshow(c, f'channel {n}')

    ch = channels[process_channel]
    logger.debug(f'Using channel {process_channel} for bgmask detection')
    retval, bgmask = cv2.threshold(ch, 150, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    logger.debug(f'OTSU-computed threshold: {retval}')
    bgmask = cv2.dilate(bgmask, imu.misc.get_kernel(3),
                iterations=1,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0)

    debug_imshow(bgmask, 'initial bgmask')
    return bgmask

def find_bgmask_thresh(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, bgmask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_TRIANGLE)
    logger.debug(f'OTSU-computed threshold: {retval}')
    debug_imshow(bgmask, 'initial bgmask')
    return bgmask

def bgmask_to_bbox(bg_mask: np.ndarray) -> tuple:
    # Find object contours
    contours, hierarchy = cv2.findContours(bg_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):
        raise Exception('ERROR: no objects detected')
    logger.debug(f'{len(contours)} contours detected')

    if is_debug():
        canvas = np.full(bg_mask.shape, 0, dtype=np.uint8)
        for i in range(len(contours)):
            cv2.drawContours(canvas, contours, i, 255, 1, hierarchy=hierarchy)
        debug_imshow(canvas, 'contours')

    # Get the maximum area contour (this is going to be an object)
    areas = [cv2.contourArea(c) for c in contours]
    contour = contours[np.argmax(areas)]

    # Find bounding box
    bbox = cv2.boundingRect(contour)
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy), bbox, contour

def replace_color(img, replace_clr, src_clr = None, mask = None, exclude_contour = None):
    if mask is None and src_clr is None:
        raise ValueError('No search parameters specified')
    if mask is None:
        img[ img == src_clr ] = replace_clr
    else:
        if exclude_contour is not None:
            # Draw the contor onto the mask with white color and slightly extend its boundaries
            mask = mask.copy()
            mask = cv2.drawContours(mask, [exclude_contour], 0, 255, -1)
            mask = cv2.dilate(mask, imu.misc.get_kernel(3),
                        iterations=1,
                        borderType=cv2.BORDER_CONSTANT,
                        borderValue=255)
            debug_imshow(mask, 'contour filled')
        img[ mask == 0 ] = replace_clr
    return img

def extract_object(img1, 
    method='multichannel', 
    extract_roi=True, 
    roi_size=None,
    bbox_relax=None,
    replace_bgclr=None,
    **kwargs):
    """ Detect and extract an object """

    if replace_bgclr is None and not extract_roi:
        raise ValueError('Invalid parameter combination ')
    if roi_size is not None and bbox_relax is not None:
        raise ValueError('Invalid parameter combination')
    if bbox_relax is None:
        bbox_relax = 0.3

    # Prepare background subtraction mask
    match method:
        case 'multichannel':
            bgmask = find_bgmask_multichannel(img1, **kwargs)
        case 'luminosity':
            bgmask = find_bgmask_luminosity(img1, **kwargs)
        case 'threshold':
            bgmask = find_bgmask_thresh(img1, **kwargs)
        case _:
            raise ValueError('Unknown method %s', method)

    # Convert to bounding box
    center, bbox, contour = bgmask_to_bbox(bgmask)
    logger.info(f'Object detected at {center} bbox {bbox}')

    if roi_size is None:
        # Relax bbox by given boundary enlargement coefficient or constant value
        if isinstance(bbox_relax, float):
            dw, dh = int(bbox[2]*bbox_relax), int(bbox[3]*bbox_relax)
        else:
            dw, dh = bbox_relax, bbox_relax

        bbox_roi = [
            max(bbox[0] - dw, 0),
            max(bbox[1] - dh, 0),
            min(bbox[0] + bbox[2] + dw, img1.shape[1]),
            min(bbox[1] + bbox[3] + dh, img1.shape[0])
        ]
    else:
        # Exact ROI size provided
        x, y = max(int(center[0] - roi_size[1]/2), 0), max(int(center[1] - roi_size[0]/2), 0)
        bbox_roi = [
            x,
            y,
            int(x + roi_size[1]),
            int(y + roi_size[0])
        ]
        if bbox_roi[2] > img1.shape[1]:
            logger.warning(f'Image is not wide enought to extract ROI centered at {center} given required width {roi_size[0]}')
            bbox_roi[2] = img1.shape[1]
        if bbox_roi[3] > img1.shape[0]:
            logger.warning(f'Image is not long enought to extract ROI centered at {center} given required height {roi_size[0]}')
            bbox_roi[3] = img1.shape[0]
    logger.info(f'ROI bbox: {bbox_roi}')

    if is_debug():
        debug_img = img1.copy()
        cv2.rectangle(debug_img, 
            (bbox_roi[0], bbox_roi[1]), 
            (bbox_roi[2], bbox_roi[3]),
            (0, 255, 0), 2)
        debug_imshow(debug_img, 'bbox')

    if replace_bgclr is not None:
        # Replace background with single color
        img1 = img1.copy()
        img1 = replace_color(img1, replace_bgclr, mask=bgmask, exclude_contour=contour)

    if extract_roi:
        # Extract ROI
        img1 = imu.get_image_area(img1, bbox_roi)

    return img1

target_dir = Path('./images.white')
for fn in Path('./images').rglob('*.png'):
    print(fn, end=' ')
    source = cv2.imread(str(fn))
    target = extract_object(source, 
                        method='luminosity',
                        extract_roi=False, 
                        replace_bgclr=imu.COLOR_WHITE)
    new_fn = target_dir if str(fn.parent) in ['images', '.'] else target_dir.joinpath(fn.parent.name)
    new_fn.mkdir(exist_ok=True)
    new_fn = new_fn.joinpath(fn.name)
    print('-->', new_fn)
    cv2.imwrite(str(new_fn), target)
