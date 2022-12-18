# LEGO sorter project
# Global flags
# (c) kol, 2022

import os
from absl import flags

flags.DEFINE_bool('gray', True, short_name='g', help='Use grayscaled images')
flags.DEFINE_bool('edges', False, short_name='e', help='Use edge-highlited images')

IMAGE_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..', 'images'))
""" Images root"""

CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..', 'checkpoints'))
""" Checkpoints root"""

BATCH_SIZE = 32
""" Batch size """

IMAGE_SIZE = (256, 256)
""" Size of images in processing pipeline """