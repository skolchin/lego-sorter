# LEGO sorter project
# GRAY2RGB conversion tests
# (c) kol, 2022

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from root_dir import OUTPUT_DIR

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


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

file_name = os.path.join(OUTPUT_DIR, '3003_test.png')
image = tf.image.decode_image(tf.io.read_file(file_name))
images = tf.convert_to_tensor([image])
images = tf.image.convert_image_dtype(images, tf.float32)
print(images.shape, images.dtype)
_ = plt.subplot(2, 2, 1)
plt.imshow(images[0])
plt.axis('off')

gray_images = tf.image.rgb_to_grayscale(images)
print(gray_images.shape, gray_images.dtype)
_ = plt.subplot(2, 2, 2)
plt.imshow(gray_images[0])
plt.axis('off')

shape = gray_images.shape
if len(shape) == 3: sobel_images = tf.expand_dims(gray_images, 0)
grad_components = tf.image.sobel_edges(gray_images)
grad_mag_components = grad_components**2
grad_mag_square = tf.math.reduce_sum(grad_mag_components,axis=-1)
sobel_images = tf.sqrt(grad_mag_square)
if len(shape) == 3: sobel_images = sobel_images[0]
print(sobel_images.shape, sobel_images.dtype)
_ = plt.subplot(2, 2, 3)
plt.imshow(sobel_images[0])
plt.axis('off')

comb_images = tf.concat([gray_images, sobel_images, sobel_images], axis=-1)
comb_images = tf.clip_by_value(comb_images, 0.0, 1.0)
print(comb_images.shape, comb_images.dtype)
_ = plt.subplot(2, 2, 4)
plt.imshow(comb_images[0])
plt.axis('off')

plt.show()
