# Image processing functions
# (c) kol, 2019-2022

""" Misc routines """

import cv2
import numpy as np
from typing import Iterable
from typing import Union, Tuple, Optional

def resize(
    img: np.ndarray, 
    new_size: Union[Iterable[int], int] = None,
    scale: Union[Iterable[float], float] = None, 
    return_extra: bool = False,
) -> Union[np.ndarray, Tuple]:

    """ Proportionally resizes an image either to specified scale or to specified size.

    Args:
        img:        An OpenCV image
        new_size:   New size. If is an iterable with 2 elements, 
                    specifies precise target size (height, width). Otherwise, 
                    specfies maximum size of any image side after resizing - in this case
                    image is resized proportionally
        scale:      Scaling ratio. If is an iterable with 2 elements, 
                    specifies image scaling ratio on (height, width). Otherwise,
                    specifies single scale for both sides
        return_extra: If True, returns actual scale along with the image

    Returns:
        If `return_extra` is False, returns only resized OpenCV image. 
        Else, returns 2-element tuple containing resized image and actuals scaling factor (tuple of 2 floats)
    """

    def _inner_resize(img, new_size, scale):
        if scale is not None:
            # Resizing by scale
            if not isinstance(scale, Iterable):
                im_scale = (scale, scale)
            else:
                if len(scale) < 2:
                    raise ValueError(f'Scale must be either scalar or 2-element vector')
                im_scale = tuple(scale[:2])

            im = cv2.resize(img, dsize=None, fx=float(im_scale[1]), fy=float(im_scale[0]))
            return im, im_scale

        else:
            # Resizing to new_size
            if isinstance(new_size, Iterable):
                # Size vector provided
                if len(new_size) < 2:
                    raise ValueError(f'New_size must be either scalar or 2-element vector')
                h, w = img.shape[:2]
                im_scale = (new_size[0] / h, new_size[1] / w)
            else:
                # Only max size given
                im_size_max = np.max(img.shape[:2])
                im_size_min = np.min(img.shape[:2])
                im_scale = float(new_size) / float(im_size_min)

                if np.round(im_scale * im_size_max) > new_size:
                    im_scale = float(new_size) / float(im_size_max)

                new_size = (new_size, new_size)
                im_scale = (im_scale, im_scale)

            im = cv2.resize(img, dsize=None, fx=im_scale[1], fy=im_scale[0])
            return im, im_scale

    if new_size is None and scale is None:
        raise ValueError('Either new_size or scale must be provided')

    result = _inner_resize(img, new_size, scale)
    return result if return_extra else result[0]

def imshow(img: np.ndarray, title: str = 'imshow', max_size: Iterable = None):
    """ Shows the image and waits for keypress.

    Args:
        img: An OpenCV image
        title: Window title. If set to one's previously used, then
            it will replace content of that window, otherwise a new window will be displayed
        max_size:   Maximum image size (height, width). 
            If actual image is bigger and may not fit to screen, it will be downsized to given one

    Returns:
        None
    """
    if img is None:
        return
    if max_size is not None and (img.shape[0] > max_size[0] or img.shape[1] > max_size[1]):
        scale_x, scale_y = max_size[1] / img.shape[1], max_size[0] / img.shape[0]
        img = resize(img, scale=min(scale_x, scale_y))
        
    cv2.imshow(title, img)
    cv2.waitKey(0)

def get_bgsub_mask(
    img: np.ndarray,
    img_bg: np.ndarray,
    kernel_size: Optional[int] = 10
) -> np.ndarray:
    """ Calculate background subtraction mask using static background and foreground images.

    Based on https://stackoverflow.com/questions/25617252/opencv-background-segmentation-subtraction-from-single-image

    Args:
        img:    An OpenCV image with background and probably some foreground objects
        img_bg: An OpenCV image with pure background

    Returns:
        A mask to subtract the background from foreground image. Use `apply_image_mask` to actually 
        extract the foreground

    Examples:

            img = cv2.imread('image.png')
            img_bg = cv2.imread('background.png')
            mask = get_bgsub_mask(img_fg, img_bg, kernel_size=21)
            masked_img = apply_image_mask(img, mask)
            cv2.imshow('Masked image', masked_img)
    """

    backSub = cv2.createBackgroundSubtractorMOG2()
    _ = backSub.apply(img_bg)
    mask = backSub.apply(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel)

    return mask_morph

def get_image_area(img: np.ndarray, area: Iterable) -> np.ndarray:
    """Get part of an image defined by rectangular area.

    Args:
        img:    An OpenCV image
        area:   Area to extract (list or tuple [x1,y1,x2,y2])

    Returns:
        Extracted area copy as OpenCV image
    """
    if not isinstance(area, Iterable) or len(area) < 4:
       raise ValueError(f'4-element iterable is expected, {type(area)} found')
    if any([a < 0 for a in area]):
       raise ValueError(f'Invalid area: {area}')
    dx = area[2] - area[0]
    dy = area[3] - area[1]
    if dx <= 0 or dy <= 0:
       raise ValueError(f'Invalid area length or width: {area}')

    if len(img.shape) > 2:
       im = np.empty((dy, dx, img.shape[2]), dtype=img.dtype)
    else:
       im = np.empty((dy, dx), dtype=img.dtype)

    im[:] = img[area[1]:area[3], area[0]:area[2]]
    return im

def zoom_at(img: np.ndarray, zoom: float, pad_color: Tuple[int] = (0,0,0)) -> np.ndarray:
    """ Zooms image

    Args:
        img:    An 1- or 3-channel OpenCV image
        zoom:   Zoom factor, float value greater than 0. 
            If it is greater than 1, then image is zoomed in (become larger), 
            and if less - zoomed out (become smaller).
        pad_color:  padding color

    Returns:
        An OpenCV image
    """
    cy, cx = [ i // 2 for i in img.shape[:-1] ]

    rot_mat = cv2.getRotationMatrix2D((cx,cy), 0, zoom)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_NEAREST, 
        borderMode=cv2.BORDER_CONSTANT, borderValue=pad_color)
