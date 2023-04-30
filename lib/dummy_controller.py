# LEGO sorter project
# Dummy controller
# (c) lego-sorter team, 2022-2023

import logging
from typing import Mapping

_logger = logging.getLogger('lego-sorter')

class DummyController:
    """ Dummy Controller 
    """
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DummyController, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.port = self.find_controller()

    @staticmethod
    def get_ports() -> Mapping:
        return ['dummy']

    @staticmethod
    def find_controller():
        return DummyController.get_ports()[0]

    def start_processing(self):
        _logger.info("Processing started")

    def stop_processing(self):
        _logger.info("Processing stopped")

    def change_state(self, state: chr, bucket: chr = ''):
        pass

    def connect(self):
        _logger.info('Controller connected')

    def wait(self):
        _logger.info('Wait state set')

    def clean(self):
        _logger.info('Clean state set')

    def run(self):
        _logger.info('Run state set')

    def select(self, bucket: chr):
        _logger.info(f'Selection state set with bucket {bucket}')

    def recognize(self):
        _logger.info('Recognition state set')

    def disconnect(self):
        _logger.info('Controller disconnected')

    def get_next_message(self):
        return None