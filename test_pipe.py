# LEGO sorter project
# Image processing pipeline tests
# (c) kol, 2022

import cv2
import logging
import numpy as np
import img_utils22 as imu

logging.basicConfig(format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def debug_imshow(img: np.ndarray, title: str = None):
    if logger.getEffectiveLevel() == logging.DEBUG:
        imu.imshow(img, title)

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
    _, bgmask = cv2.threshold(ch, 150, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bgmask = cv2.dilate(bgmask, imu.misc.get_kernel(3),
                iterations=1,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0)

    debug_imshow(bgmask, 'initial bgmask')
    return bgmask

def bgmask_to_bbox(bg_mask: np.ndarray) -> tuple:
    # Find object contours
    contours, _ = cv2.findContours(bg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):
        raise Exception('ERROR: no objects detected')

    # Get the maximum area contour (this is going to be an object)
    areas = [cv2.contourArea(c) for c in contours]
    contour = contours[np.argmax(areas)]

    # Find bounding box
    bbox = cv2.boundingRect(contour)
    # M = cv2.moments(contour)
    # cx = int(M['m10'] / M['m00'])
    # cy = int(M['m01'] / M['m00'])
    return bbox, contour

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
        img[ mask == 0 ] = replace_clr
    return img

def extract_object(img1, method='multichannel', extract_roi=True, enlarge_boundaries=0.3, replace_bgclr=None):
    """ Detect and extract an object """

    if replace_bgclr is None and not extract_roi:
        raise ValueError('Invalid parameter combination')

    # Prepare background subtraction mask
    if method == 'multichannel':
        bgmask = find_bgmask_multichannel(img1)
    elif method == 'luminosity':
        bgmask = find_bgmask_luminosity(img1)
    else:
        raise ValueError('Unknown method %s', method)

    # Convert to bounding box
    bbox, contour = bgmask_to_bbox(bgmask)
    logger.info(f'Object detected: {bbox}')

    # Relax bbox by given boundary enlargement coefficient or value
    if isinstance(enlarge_boundaries, float):
        dw, dh = int(bbox[2]*enlarge_boundaries), int(bbox[3]*enlarge_boundaries)
    else:
        dw, dh = int(enlarge_boundaries), int(enlarge_boundaries)

    bbox_relaxed = [
        max(bbox[0] - dw, 0),
        max(bbox[1] - dh, 0),
        min(bbox[0] + bbox[2] + dw, img1.shape[1]),
        min(bbox[1] + bbox[3] + dh, img1.shape[0])
    ]
    if logger.getEffectiveLevel() == logging.DEBUG:
        debug_img = img1.copy()
        cv2.rectangle(debug_img, 
            (bbox_relaxed[0], bbox_relaxed[1]), 
            (bbox_relaxed[2], bbox_relaxed[3]),
            (0, 255, 0), 2)
        imu.imshow(debug_img, 'bbox')

    if replace_bgclr is not None:
        # Replace background with single color
        img1 = img1.copy()
        img1 = replace_color(img1, replace_bgclr, mask=bgmask, exclude_contour=contour)

    if extract_roi:
        # Extract ROI
        return imu.get_image_area(img1, bbox_relaxed)

    return img1

def main():
    # source = cv2.imread('out\\3003_test.png')
    # imu.imshow(source, 'source')
    # result = extract_object(source, method='luminosity')
    # imu.imshow(result, 'result')
    # (314, 273, 204, 212)

    source = cv2.imread('out\\photo_2022-12-11_21-48-19.jpg')
    # source = imu.resize(source, (815,815))    
    imu.imshow(source, 'source')
    result = extract_object(source, method='multichannel', extract_roi=False, replace_bgclr=imu.COLOR_WHITE)
    imu.imshow(result, 'result')
    cv2.imwrite(f'out\\3003_test2.png', result)

if __name__ == '__main__':
    main()
