# LEGO sorter project
# Edges detection pipeline tests
# (c) lego-sorter team, 2022-2025

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from lib.globals import OUT_DIR

# import cv2
# import img_utils22 as imu

# def main():
#     pipe1 = imu.Pipe() | imu.LoadFile('./out/3003_test.png') | imu.Resize(scale=0.5) | imu.ShowImage('source')
#     img1 = pipe1(None)

#     pipe2 = imu.Pipe() | imu.EqualizeLuminosity() | imu.Blur() | imu.Gray() | imu.Edges() | imu.Dilate(kernel_size=1) | imu.ShowImage('mask')
#     mask = pipe2(img1)
#     mask_inv = cv2.bitwise_not(mask)

#     img2 = cv2.bitwise_and(img1, img1, mask=mask_inv)
#     img_fg = np.full(img1.shape, imu.COLOR_BLACK, img1.dtype)
#     img_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
#     img2 = cv2.add(img2, img_fg)
#     # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     imu.ShowImage('merged')(img2)

# https://stackoverflow.com/questions/42835247/how-to-implement-histogram-equalization-for-images-in-tensorflow
def tf_equalize_histogram(image):
    values_range = tf.constant([0., 255.], dtype = tf.float32)
    histogram = tf.histogram_fixed_width(tf.cast(image, tf.float32), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(image)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(tf.cast(cdf - cdf_min, tf.float32) * 255. / tf.cast(pix_cnt - 1, tf.float32))
    px_map = tf.cast(px_map, tf.uint8)

    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
    return eq_hist

def main():
    image_raw = tf.io.read_file(os.path.join(OUT_DIR,'3003_test.png'))
    image = tf.image.decode_image(image_raw)

    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.rgb_to_grayscale(image)
    # image = tf_equalize_histogram(image)
    image = tf.cast(image, tf.float32)

    grad_components = tf.image.sobel_edges(tf.expand_dims(image, 0))
    grad_mag_components = grad_components**2
    grad_mag_square = tf.math.reduce_sum(grad_mag_components,axis=-1) # sum all magnitude components
    grad_mag_img = tf.sqrt(grad_mag_square) # this is the image tensor you want

    _, axs = plt.subplots(1,2)
    axs[0].set_title('Source')
    axs[0].imshow(image.numpy().astype('uint8'))
    axs[0].axis('off')
    axs[1].set_title('Edges map')
    axs[1].imshow(grad_mag_img[0].numpy().astype('uint8'))
    axs[1].axis('off')
    plt.show()

if __name__ == '__main__':
    main()
