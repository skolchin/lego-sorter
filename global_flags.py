# LEGO sorter project
# Global flags
# (c) kol, 2022

from absl import flags

flags.DEFINE_bool('gray', False, short_name='g', help='Train on/use grayscale images')
flags.DEFINE_bool('edges', False, short_name='e', help='Train on/use edge-highlited images')