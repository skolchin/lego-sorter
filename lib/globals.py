# LEGO sorter project
# Global flags
# (c) lego-sorter team, 2022-2025

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
""" Root dir """

IMAGE_DIR = ROOT_DIR / 'images'
""" Images root """

MODELS_DIR = ROOT_DIR / 'models'
""" Models root """

CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'
""" Checkpoints root """

OUTPUT_DIR = ROOT_DIR / 'out'
""" Output dir """

RETRAIN_DIR = ROOT_DIR / 'retrain'
""" Retrain images dir """

BATCH_SIZE = 32
""" Batch size """
