# LEGO sorter project
# Global flags
# (c) kol, 2022

import os
from absl import flags

flags.DEFINE_bool('gray', False, short_name='g', 
    help='Convert images to grayscale')
flags.DEFINE_bool('edges', False, short_name='e', 
    help='Convert images to wireframe (edges-only) images')
flags.DEFINE_bool('emboss', False, short_name='x', 
    help='Mix wireframe with actual image, can be combined with --gray')
flags.DEFINE_boolean('zoom', False, short_name='z', 
    help='Apply zoom augmentation (slows down the training by x5)')

ROOT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
""" Root dir """

IMAGE_DIR = os.path.join(ROOT_DIR, 'images')
""" Images root """

CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
""" Checkpoints root """

OUTPUT_DIR = os.path.join(ROOT_DIR, 'out')
""" Output dir """


BATCH_SIZE = 32
""" Batch size """

IMAGE_SIZE = (256, 256)
""" Size of images in processing pipeline """