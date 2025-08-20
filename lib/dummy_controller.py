# LEGO sorter project
# Dummy controller
# (c) lego-sorter team, 2022-2025

import logging
from typing import Mapping

_logger = logging.getLogger(__name__)

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
    def find_controller() -> str:
        return 'dummy'

    def start_processing(self):
        _logger.info("Processing started")

    def stop_processing(self):
        _logger.info("Processing stopped")

    def change_state(self, state: str, bucket: str = '', confirm: bool = True):
        pass

    def connect(self):
        _logger.info('Controller connected')

    def wait(self):
        _logger.info('Wait state set')

    def clean(self):
        _logger.info('Clean state set')

    def run(self):
        _logger.info('Run state set')

    def select(self, bucket: str):
        _logger.info(f'Selection state set with bucket {bucket}')

    def recognize(self):
        _logger.info('Recognition state set')

    def disconnect(self):
        _logger.info('Controller disconnected')

    def get_next_message(self):
        return None