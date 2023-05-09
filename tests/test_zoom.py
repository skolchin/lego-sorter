# LEGO sorter project
# Zoom test
# (c) lego-sorter team, 2022-2023

import os
import cv2
import numpy as np
import img_utils22 as imu
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from absl import app
from root_dir import ROOT_DIR

def draw_ruler(img, x=50, y=-1):
    w = img.shape[1]-2*x
    y = y if y >=0 else img.shape[0] // 2
    cv2.line(img, (x,y), (w+x, y), imu.COLOR_BLACK, 2)
    for n in np.linspace(0, 100, 11):
        cv2.line(img, (x,y-5), (x, y+5), imu.COLOR_BLACK, 1)
        cv2.putText(img, f'{int(n)}', (x-5,y+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, imu.COLOR_BLACK)
        x += int(w/10)

# https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv
def zoom_at(img, zoom, pad_color=imu.COLOR_BLACK):
    cy, cx = [ i //2 for i in img.shape[:-1] ]

    rot_mat = cv2.getRotationMatrix2D((cx,cy), 0, zoom)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_NEAREST, 
        borderMode=cv2.BORDER_CONSTANT, borderValue=pad_color)

def zoom_scale(img, zoom):
    img = imu.rescale(img, scale=zoom, center=True, pad_color=imu.COLOR_WHITE)
    # size = img.shape[:2]
    # new_size = (int(img.shape[1] * zoom), int(img.shape[0] * zoom))
    # img, _ = center_image(img, new_size)
    # img = imu.resize(img, new_size=size)
    return img

def zoom_tf(img, zoom, pad_color=imu.COLOR_BLACK):
    from lib.image_dataset import zoom_image
    return zoom_image(img, zoom, pad_color[0])

def main(_):
    original = np.full((200, 640, 3), imu.COLOR_WHITE, np.uint8)
    draw_ruler(original)

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.title('zoom')
    plt.axis('off')
    fig.tight_layout()

    zoom = 1.0
    def apply_zoom(val):
        nonlocal zoom
        zoom = val

        zoomed = np.full((200, 640, 3), imu.COLOR_WHITE, np.uint8)
        draw_ruler(zoomed)
        zoomed = zoom_at(zoomed, zoom, imu.COLOR_WHITE)
        merged = np.vstack((original, zoomed))
        ax.imshow(merged)
        ax.set_title(f'{zoom:.4f}')
        fig.canvas.draw_idle()

    slider = Slider(
        ax=fig.add_axes([0.1, 0.25, 0.0225, 0.63]),
        label='zoom',
        valmin=0.1,
        valmax=3.0,
        valstep=0.1,
        valinit=zoom,
        orientation="vertical"
    )
    slider.on_changed(apply_zoom)
    apply_zoom(zoom)
    plt.show()

if __name__ == '__main__':
    app.run(main)