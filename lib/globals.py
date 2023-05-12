# LEGO sorter project
# Global flags
# (c) lego-sorter team, 2022-2023

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
""" Root dir """

IMAGE_DIR = os.path.join(ROOT_DIR, 'images')
""" Images root """

CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
""" Checkpoints root """

OUTPUT_DIR = os.path.join(ROOT_DIR, 'out')
""" Output dir """

RETRAIN_DIR = os.path.join(ROOT_DIR, 'retrain')
""" Retrain images dir """

BATCH_SIZE = 32
""" Batch size """

IMAGE_SIZE = (224, 224)
""" Size of images in processing pipeline """
