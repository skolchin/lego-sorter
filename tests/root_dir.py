import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
OUT_DIR = os.path.join(ROOT_DIR, 'out')
sys.path.append(ROOT_DIR)
