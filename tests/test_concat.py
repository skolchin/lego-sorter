# LEGO sorter project
# GRAY2RGB conversion tests
# (c) lego-sorter team, 2022-2025

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from lib.globals import ROOT_DIR, OUTPUT_DIR

# from tensorflow.python.framework import ops
# from tensorflow.python.ops import array_ops
# def merge(images, name=None):
#   with ops.name_scope(name, 'grayscale_to_rgb', [images]) as name:

#     images = ops.convert_to_tensor(images, name='images')
#     rank_1 = array_ops.expand_dims(array_ops.rank(images) - 1, 0)
#     shape_list = ([array_ops.ones(rank_1, dtype=dtypes.int32)] +
#                   [array_ops.expand_dims(3, 0)])
#     multiples = array_ops.concat(shape_list, 0)
#     rgb = array_ops.tile(images, multiples, name=name)
#     rgb.set_shape(images.get_shape()[:-1].concatenate([3]))
#     return rgb

# file_name = os.path.join(OUTPUT_DIR, '3003_test.png')
file_name = os.path.join(ROOT_DIR, 'images', '3003', '3003_0.png')

image = tf.image.decode_image(tf.io.read_file(file_name), channels=3)
images = tf.convert_to_tensor([image])
images = tf.image.convert_image_dtype(images, tf.float32)
print(images.shape, images.dtype)
_ = plt.subplot(2, 3, 1)
plt.title('Source')
plt.imshow(images[0])
plt.axis('off')

gray_images = tf.image.rgb_to_grayscale(images)
print(gray_images.shape, gray_images.dtype)
_ = plt.subplot(2, 3, 2)
plt.title('Gray')
plt.imshow(gray_images[0], cmap='gray')
plt.axis('off')

shape = gray_images.shape
if len(shape) == 3: sobel_images = tf.expand_dims(gray_images, 0)
grad_components = tf.image.sobel_edges(gray_images)
grad_mag_components = grad_components**2
grad_mag_square = tf.math.reduce_sum(grad_mag_components,axis=-1)
sobel_images = tf.sqrt(grad_mag_square)
if len(shape) == 3: sobel_images = sobel_images[0]
print(sobel_images.shape, sobel_images.dtype)
print(np.unique(sobel_images, return_counts=True))
_ = plt.subplot(2, 3, 3)
plt.title('Edges')
plt.imshow(sobel_images[0], cmap='gray')
plt.axis('off')

mask = tf.where(sobel_images < 0.5, 0.0, 1.0)
kernel = tf.zeros((3, 3, mask.get_shape()[3])) 
mask = tf.nn.dilation2d(mask, kernel, (1,1,1,1), 'SAME', 'NHWC', (1,2,2,1), 'dilation')
print(np.unique(mask, return_counts=True))
_ = plt.subplot(2, 3, 4)
plt.title('Mask')
plt.imshow(mask[0], cmap='gray')
plt.axis('off')

# comb_images = tf.concat([sobel_images, gray_images, gray_images], axis=-1)
# comb_images = tf.clip_by_value(comb_images, 0.0, 1.0)
comb_images = tf.where(tf.equal(mask, 0.0), images, tf.constant((1.0, 0.0, 0.0), images.dtype))
print(comb_images.shape, comb_images.dtype)
_ = plt.subplot(2, 3, 5)
plt.title('Result')
plt.imshow(comb_images[0])
plt.axis('off')

plt.show()
