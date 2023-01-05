# LEGO sorter project
# HW controller prototype
# (c) kol, 2022

import os
import json
import logging
from typing import Tuple, Iterable

from .globals import ROOT_DIR

logger = logging.getLogger(__name__)

class Controller:
    """ Simple HW controller prototype """

    UNKNOWN: str = '_'

    def __init__(self):
        with open(os.path.join(ROOT_DIR, 'lib', 'bins.json'), 'r') as fp:
            self.bins = json.load(fp)
            if not self.UNKNOWN in self.bins:
                self.bins[self.UNKNOWN] = (-1, -1)

    @property
    def labels(self) -> Iterable[str]:
        return list(set(self.bins.keys()).difference([self.UNKNOWN]))

    def find_bin(self, label: str) -> Tuple[str]:
        try:
            return tuple(self.bins[label])
        except KeyError:
            return tuple(self.bins[self.UNKNOWN])

    def start(self):
        logger.info('Controller started')

    def stop(self):
        logger.info('Controller stopped')

    def process_object(self, label: str):
        p = self.find_bin(label)
        logger.info(f'Moving "{label}" object to {p} bin')
