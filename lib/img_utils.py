# Image processing functions
# (c) kol, 2019-2022

""" Misc routines """

import cv2
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from typing import Sequence
from typing import Union, Tuple, Optional

ColorT = Tuple[int,int,int]
SizeT = Tuple[int,int]
BBoxT = Tuple[int,int,int,int]
ImageT = npt.NDArray

COLOR_BLACK: ColorT = (0,0,0)
COLOR_RED: ColorT = (0,0,255)
COLOR_BLUE: ColorT = (255,0,0)
COLOR_GREEN: ColorT = (0,255,0)
COLOR_GRAY: ColorT = (127,127,127)
COLOR_WHITE: ColorT = (255,255,255)

def rgb_to_rgba(rgb: ImageT, alpha: float = 1.0) -> ImageT:
    """ Convert RGB color to RGBA color

    Args:
        rgb:    3-element numpy array or compatible type specifying a color to convert
        alpha:  Alpha channel value

    Returns:
        4-element numpy array with RGBA color
    """
    if not isinstance(rgb, np.ndarray):
        rgb = np.array(rgb)
    assert rgb.shape[-1] == 3

    rgba = np.full((rgb.shape[0], rgb.shape[1], 4), alpha, dtype=rgb.dtype)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    rgba[:,:,0] = r
    rgba[:,:,1] = g
    rgba[:,:,2] = b

    return rgba

def resize(
    img: ImageT, 
    new_size: Union[SizeT, int, None] = None,
    scale: Union[Sequence[float], float, None] = None, 
    return_extra: bool = False,
) -> Union[ImageT, Tuple]:

    """ Proportionally resizes an image either to specified scale or to specified size.

    Args:
        img:        An OpenCV image
        new_size:   New size. If is an Sequence with 2 elements, 
                    specifies precise target size (height, width). Otherwise, 
                    specfies maximum size of any image side after resizing - in this case
                    image is resized proportionally
        scale:      Scaling ratio. If is an Sequence with 2 elements, 
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
            if not isinstance(scale, Sequence):
                im_scale = (scale, scale)
            else:
                if len(scale) < 2:
                    raise ValueError(f'Scale must be either scalar or 2-element vector')
                im_scale = tuple(scale[:2])

            im = cv2.resize(img, dsize=None, fx=float(im_scale[1]), fy=float(im_scale[0]))
            return im, im_scale

        else:
            # Resizing to new_size
            if isinstance(new_size, Sequence):
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

def patch(
    img: ImageT, 
    x: int, 
    y: int, 
    patch_img: ImageT, 
    patch_mask: Optional[ImageT] = None, 
    clip: bool = False, 
    alpha: Optional[float] = None) -> ImageT:

    """ Applies a patch at given coordinates with with optional masking and alpha-channel blending.

    Args:
        img:        A 3-channel OpenCV image
        x:          Patch X offset
        y:          Patch Y offset
        patch_img:  Patch OpenCV image
        patch_mask: Optional boolean mask to apply over patch
        clip:       Whether to clip patch if it's bigger than image
        alpha:      Alpha channel value

    Returns:
        An OpenCV image
    """

    if y >= img.shape[0] or x >= img.shape[1] or y < 0 or x < 0:
        raise ValueError("Invalid coordinates")

    h, w = patch_img.shape[:2]
    if (y + h > img.shape[0] or x + w > img.shape[1]) and not clip:
        raise ValueError("Patch is outside image area and clipping not specified")
    if y + h >= img.shape[0] and clip:
        h = img.shape[0] - y
    if x + w >= img.shape[1] and clip:
        w = img.shape[1] - x
        patch_img = patch_img[0:h, 0:w]
        patch_mask = patch_mask[0:h, 0:w] if patch_mask is not None else None

    if patch_img.shape[-1] == 3 and img.shape[-1] == 4:
        # Patch contains alpha channel
        patch_img = rgb_to_rgba(patch_img)

    dst = img.copy()
    area = dst[y:(y+h), x:(x+w)]
    if patch_mask is None:
        if alpha is None:
            area = patch_img
        else:
            cv2.addWeighted(area, 1 - alpha, patch_img, alpha, 0, area)
    else:
        if alpha is None:
            area[patch_mask > 0] = patch_img[patch_mask > 0]
        else:
            dtyp = area.dtype
            a = area[patch_mask > 0] * (1-alpha)
            p = patch_img[patch_mask > 0] * alpha
            area[patch_mask > 0] = a.astype(dtyp) + p.astype(dtyp)

    dst[y:(y+h), x:(x+w)] = area
    return dst

def imshow(img: ImageT, title: str = 'imshow', max_size: Tuple[int, int] | None = None):
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
        img = resize(img, scale=min(scale_x, scale_y))   # pyright: ignore[reportAssignmentType]
        
    cv2.imshow(title, img)
    cv2.waitKey(0)

def get_bgsub_mask(
    img: ImageT,
    img_bg: ImageT,
    kernel_size: Optional[int] = 10
) -> ImageT:
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

    sz: int = kernel_size or 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(sz,sz))
    mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel)

    return mask_morph

def get_image_area(img: ImageT, area: Sequence) -> ImageT:
    """Get part of an image defined by rectangular area.

    Args:
        img:    An OpenCV image
        area:   Area to extract (list or tuple [x1,y1,x2,y2])

    Returns:
        Extracted area copy as OpenCV image
    """
    if not isinstance(area, Sequence) or len(area) < 4:
       raise ValueError(f'4-element Sequence is expected, {type(area)} found')
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

def zoom_at(img: ImageT, zoom: float, pad_color: ColorT = COLOR_BLACK) -> ImageT:
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

def _image_dimensions(image, rank):
  """Returns the dimensions of an image tensor.

  Origin: https://github.com/tensorflow/models/blob/master/research/deeplab/core/preprocess_utils.py#L68

  Args:
    image: A rank-D Tensor. For 3-D of shape: `[height, width, channels]`.
    rank: The expected rank of the image

  Returns:
    A list of corresponding to the dimensions of the input image. Dimensions
      that are statically known are python integers, otherwise they are integer
      scalar tensors.
  """
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(image), rank)
    return [
        s if s is not None else d for s, d in zip(static_shape, dynamic_shape)  #type:ignore
    ]


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
  """Pads the given image with the given pad_value.

  Works like tf.image.pad_to_bounding_box, except it can pad the image
  with any given arbitrary pad value and also handle images whose sizes are not
  known during graph construction.

  Origin: https://github.com/tensorflow/models/blob/master/research/deeplab/core/preprocess_utils.py

  Args:
    image: 3-D tensor with shape [height, width, channels]
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
    pad_value: Value to pad the image tensor with.

  Returns:
    3-D tensor of shape [target_height, target_width, channels].

  Raises:
    ValueError: If the shape of image is incompatible with the offset_* or
    target_* arguments.
  """

  with tf.name_scope('pad_to_bounding_box'):
    image = tf.convert_to_tensor(image, name='image')
    original_dtype = image.dtype
    if original_dtype != tf.float32 and original_dtype != tf.float64:
      # If image dtype is not float, we convert it to int32 to avoid overflow.
      image = tf.cast(image, tf.int32)
    image_rank_assert = tf.Assert(
        tf.logical_or(
            tf.equal(tf.rank(image), 3),
            tf.equal(tf.rank(image), 4)),
        ['Wrong image tensor rank.'])
    with tf.control_dependencies([image_rank_assert]):
      image -= pad_value
    image_shape = image.get_shape()
    is_batch = True
    if image_shape.ndims == 3:
      is_batch = False
      image = tf.expand_dims(image, 0)
    elif image_shape.ndims is None:
      is_batch = False
      image = tf.expand_dims(image, 0)
      image.set_shape([None] * 4)
    elif image.get_shape().ndims != 4:
      raise ValueError('Input image must have either 3 or 4 dimensions.')
    _, height, width, _ = _image_dimensions(image, rank=4)
    target_width_assert = tf.Assert(
        tf.greater_equal(
            target_width, width),
        ['target_width must be >= width'])
    target_height_assert = tf.Assert(
        tf.greater_equal(target_height, height),
        ['target_height must be >= height'])
    with tf.control_dependencies([target_width_assert]):
      after_padding_width = target_width - offset_width - width
    with tf.control_dependencies([target_height_assert]):
      after_padding_height = target_height - offset_height - height
    offset_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(after_padding_width, 0),
            tf.greater_equal(after_padding_height, 0)),
        ['target size not possible with the given target offsets'])
    batch_params = tf.stack([0, 0])
    height_params = tf.stack([offset_height, after_padding_height])
    width_params = tf.stack([offset_width, after_padding_width])
    channel_params = tf.stack([0, 0])
    with tf.control_dependencies([offset_assert]):
      paddings = tf.stack([batch_params, height_params, width_params,
                           channel_params])
    padded = tf.pad(image, paddings)
    if not is_batch:
      padded = tf.squeeze(padded, axis=[0])
    outputs = padded + pad_value
    if outputs.dtype != original_dtype:
      outputs = tf.cast(outputs, original_dtype)
    return outputs

def zoom_image(image: Union[ImageT, tf.Tensor], zoom: float,
               pad_color: int = 0) -> Union[ImageT, tf.Tensor]:
    """ Image zoom with tensorflow functions.

    Arguments:
        image:  Either OpenCV (`ImageT`) or Tensorflow (`tf.Tensor`) image array.
            The image must be of supported `dtype` (`uint8` or `float`).
        zoom:   Zoom factor, float value greater than 0.
            If it is less than 1, then image is zoomed in (become larger),
            and if greater - zoomed out (become smaller) (this is oppozite to OpenCV's zoom behaviour).
        pad_color:  padding color (0 - black, 255 - white)

    Returns:
        zoomed image of the same type and size as original
    """

    if zoom < 0.0:
        raise ValueError(f'Zoom must be greater than zero')

    if zoom == 1.0:
        # No zoom
        return image

    is_tensor = isinstance(image, tf.Tensor)
    if not is_tensor:
        shape = image.shape
        image_size = np.array(shape[:2])
    else:
        shape = image.get_shape()
        image_size = np.array(shape[:2]) if (shape.ndims or 1) < 4 else np.array(shape[1:3])

    if zoom < 1.0:
        # Zoom in (result is larger)
        crop_image = tf.image.central_crop(image, zoom)
        zoomed_image = tf.image.resize(crop_image, image_size, preserve_aspect_ratio=False)
    else:
        # Zoom out (result is smaller)
        zoom_size = (zoom * image_size).astype('int')
        offset_height, offset_width = (zoom_size[0]-image_size[0]) // 2, (zoom_size[1]-image_size[1]) // 2
        target_height, target_width = zoom_size[0], zoom_size[1]
        resized_image = pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width, pad_color)
        zoomed_image = tf.image.resize(resized_image, image_size, preserve_aspect_ratio=False)

    zoomed_image = tf.cast(zoomed_image, image.dtype)
    return zoomed_image.numpy() if not is_tensor and isinstance(zoomed_image, tf.Tensor) else zoomed_image #type:ignore
