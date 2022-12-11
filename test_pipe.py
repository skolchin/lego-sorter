# LEGO sorter project
# Overall pipeline tests
# (c) kol, 2022

import cv2
import numpy as np
import img_utils22 as imu

def imshow(img: np.ndarray, title: str = None):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)

def get_bgmask(img, bgclr):
    mask = imu.get_bgsub_mask(img, np.full(img.shape, bgclr, img.dtype))
    nonzero = np.nonzero(mask)
    if not nonzero[0].any() or not nonzero[1].any():
        # only background color
        return None
    return mask

BBOX_LAG = 50

def detect_object(filename, debug=False):
    # Load the file
    img1 = cv2.imread(filename)
    img1 = imu.resize(img1, scale=0.5)
    if debug:
        imshow(img1, 'source')

    # Find most popular color, which is assumed to be background
    bgclr = bincount_app(img1)
    print(f'Assuming background color is {bgclr}')

    # Build a background subtraction mask
    bg_mask = get_bgmask(img1, bgclr)
    if bg_mask is None:
        print('ERROR: no objects detected')
        return None
    if debug:
        imshow(bg_mask, 'initial mask')

    # Flood fill holes within the mask starting from all 4 corners
    bg_mask[ bg_mask != 0 ] = 255
    corners = [
        (0, 0), 
        (0, bg_mask.shape[0]-1), 
        (bg_mask.shape[1]-1, 0), 
        (bg_mask.shape[1]-1, bg_mask.shape[0]-1)
    ]
    for pt in corners:
        # cv2.circle(bg_mask, pt, 5, 255, -1)
        cv2.floodFill(bg_mask, None, pt, 0)
    if debug:
        imshow(bg_mask, 'filled mask')

    # Find object contours
    contours, hierarchy = cv2.findContours(bg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):
        print('ERROR: no objects detected')
        return None

    # Get the maximum area contour (this is going to be an object)
    areas = [cv2.contourArea(c) for c in contours]
    contour = contours[np.argmax(areas)]

    # Find bounding box and the center of countour
    bbox = cv2.boundingRect(contour)
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print(f'Objects bbox is {bbox} centered at ({cy,cx})')

    # Relax a bounding box with some lag
    bbox_relaxed = [
        max(bbox[0] - BBOX_LAG, 0),
        max(bbox[1] - BBOX_LAG, 0),
        min(bbox[0] + bbox[2] + BBOX_LAG, img1.shape[1]),
        min(bbox[1] + bbox[3] + BBOX_LAG, img1.shape[0])
    ]
    if debug:
        img_copy = img1.copy()
        cv2.drawContours(img_copy, [contour], 0, (0, 0, 255))
        cv2.rectangle(img_copy, 
            (bbox_relaxed[0], bbox_relaxed[1]), 
            (bbox_relaxed[2], bbox_relaxed[3]),
            (0, 255, 0), 2)
        imshow(img_copy, 'countour and bbox')

    # Remove any background replacing it with single color
    img2 = img1.copy()
    img2[ bg_mask == 0] = imu.COLOR_WHITE

    # Extract ROI
    img2 = imu.get_image_area(img2, bbox_relaxed)
    if debug:
        imshow(img2, 'result')
    return img2

def main():
    detect_object('out\\3003_test.png', debug=True)
    # detect_object('out\\photo_2022-11-07_15-01-22.jpg', debug=True)

if __name__ == '__main__':
    try:
        main()
    finally:
        cv2.destroyAllWindows()
