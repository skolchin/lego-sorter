# LEGO sorter project
# Zoom test
# (c) kol, 2022

import cv2
import numpy as np
import img_utils22 as imu
import tensorflow as tf
from typing import Iterable, Tuple

def draw_ruler(img, caption, x=50, y=80):
    w = img.shape[1]-2*x
    cv2.putText(img, caption, (x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, imu.COLOR_BLACK)
    cv2.line(img, (x,y), (w+x, y), imu.COLOR_BLACK, 2)
    for n in np.linspace(0, 100, 11):
        cv2.line(img, (x,y-5), (x, y+5), imu.COLOR_BLACK, 1)
        cv2.putText(img, f'{int(n)}', (x-5,y+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, imu.COLOR_BLACK)
        x += int(w/10)

# https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv
def zoom_at(img, zoom, angle=0, coord=None):
    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    
    rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_NEAREST, 
        borderMode=cv2.BORDER_CONSTANT, borderValue=imu.COLOR_WHITE)
    
    return result

def zoom_scale(img, zoom):
    img = imu.rescale(img, scale=zoom, center=True, pad_color=imu.COLOR_WHITE)
    # size = img.shape[:2]
    # new_size = (int(img.shape[1] * zoom), int(img.shape[0] * zoom))
    # img, _ = center_image(img, new_size)
    # img = imu.resize(img, new_size=size)
    return img

def zoom_tf(img, zoom):
    r = (0-zoom, 0-zoom) if zoom < 1 else (zoom-1, zoom-1)
    layer = tf.keras.layers.RandomZoom(r, r, fill_mode='nearest', interpolation='nearest')
    return layer(img).numpy()

def main():
    canvas = np.full((480, 640, 3), imu.COLOR_WHITE, np.uint8)
    draw_ruler(canvas, 'original', y=200)
    imu.imshow(canvas, 'canvas')

    # zoomed = zoom_scale(canvas, 0.5)
    # imu.imshow(zoomed, f'zoom: {0.5}')

    for n, zoom in enumerate([0.3, 0.5, 1.3, 1.5]):
        zoomed = zoom_tf(canvas, zoom)
        draw_ruler(zoomed, f'zoom: {zoom}', y=200+n*40)
        imu.imshow(zoomed, f'zoom: {zoom}')

if __name__ == '__main__':
    main()