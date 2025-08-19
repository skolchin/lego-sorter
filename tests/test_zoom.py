# LEGO sorter project
# Testing various zoom functions
# (c) lego-sorter team, 2022-2025

import cv2
import numpy as np
import lib.img_utils as imu
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from absl import app

def draw_ruler(img, x=50, y=-1):
    w = img.shape[1]-2*x
    y = y if y >=0 else img.shape[0] // 2
    cv2.line(img, (x,y), (w+x, y), (0,0,0), 2)
    for n in np.linspace(0, 100, 11):
        cv2.line(img, (x,y-5), (x, y+5), (0,0,0), 1)
        cv2.putText(img, f'{int(n)}', (x-5,y+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,0))
        x += int(w/10)

def zoom_tf(img, zoom, pad_color=(0,0,0)):
    from lib.image_dataset import zoom_image
    return zoom_image(img, zoom, pad_color[0])

def main(_):
    original = np.full((200, 640, 3), (255,255,255), np.uint8)
    draw_ruler(original)

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.title('zoom')
    plt.axis('off')

    def apply_zoom(zoom):
        zoomed = np.full((200, 640, 3), (255,255,255), np.uint8)
        draw_ruler(zoomed)
        zoomed = imu.zoom_at(zoomed, zoom, (255,255,255))
        merged = np.vstack((original, zoomed))
        ax.imshow(merged)
        ax.set_title(f'{zoom:.1f}')
        fig.canvas.draw_idle()

    slider = Slider(
        ax=fig.add_axes([0.1, 0.25, 0.0225, 0.63]),
        label='zoom',
        valmin=0.1,
        valmax=3.0,
        valstep=0.1,
        valinit=1.0,
        orientation='vertical'
    )
    slider.on_changed(apply_zoom)
    apply_zoom(1.0)
    plt.show()

def main_alt(_):
    from lib.image_dataset import load_dataset

    ds = load_dataset()
    plt.figure(figsize=(8, 8))
    for images, labels in ds.tfds.take(1):
        resized_images = zoom_tf(images, 1.5, (255,))
        for i in range(4):
            _ = plt.subplot(2, 2, i + 1)
            label = ds.class_names[np.argmax(labels[i])]
            image = resized_images[i].numpy()
            plt.title(f'{label}')
            plt.imshow(image.astype('uint8'))
            plt.axis('off')
    plt.show()


if __name__ == '__main__':
    app.run(main)
