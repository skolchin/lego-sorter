# LEGO sorter project
# Global flags
# (c) lego-sorter team, 2022-2025

from pathlib import Path

ROOT_DIR: Path = Path(__file__).parent.parent
""" Root dir """

IMAGE_DIR: Path = ROOT_DIR / 'images'
""" Images root """

MODELS_DIR: Path = ROOT_DIR / 'models'
""" Models root """

CHECKPOINT_DIR: Path = ROOT_DIR / 'checkpoints'
""" Checkpoints root """

OUTPUT_DIR: Path = ROOT_DIR / 'out'
""" Output dir """

RETRAIN_DIR: Path = ROOT_DIR / 'retrain'
""" Retrain images dir """

BATCH_SIZE: int = 32
""" Default batch size """
