# LEGO sorter project
# Global flags
# (c) kol, 2022

import os
from absl import flags

flags.DEFINE_bool('gray', False, short_name='g', help='Use grayscaled images')
flags.DEFINE_bool('edges', False, short_name='e', help='Use edge-highlited images')

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