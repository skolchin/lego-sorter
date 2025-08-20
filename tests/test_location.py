import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type:ignore

model = load_model('models/openblock/location_1668632717.hdf5')
model.compile()

image = tf.image.decode_image(tf.io.read_file('out/3003_test.png'), channels=3)
image = tf.cast(image, tf.float32)
image = tf.divide(image, 255.0)
cv2.imshow('Source', cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR))

resized = tf.image.resize(image, (100, 150), 'bicubic', preserve_aspect_ratio=False)
resized_batch = tf.expand_dims(resized, 0)

predictions = model.predict(resized_batch)
_, side, top = predictions

side_x = int(side[0][0] * image.shape[1])
side_y = int(side[0][1] * image.shape[0])

top_x = int(top[0][0] * image.shape[1])
top_y = int(top[0][1] * image.shape[0])

marked = image.numpy().copy()
cv2.circle(marked, (side_x, side_y), 5, (0,0,1), -1 )
cv2.circle(marked, (top_x, top_y), 5, (1,0,0), -1 )
cv2.imshow('Marked', marked)
cv2.waitKey(0)