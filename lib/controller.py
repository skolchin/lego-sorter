# LEGO sorter project
# HW controller prototype
# (c) lego-sorter team, 2022-2023

import logging
import serial
from serial.tools import list_ports
from queue import Queue, Empty
from threading import Event, Thread

class NoSorterException(Exception):
    def __init__(self):
        super().__init__("Unable to find lego sorter. Is it connected?")

_logger = logging.getLogger('lego-sorter')

class Controller:
    """ Arduino controller. Implements singleton pattern (one instance per application) """

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Controller, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.port = self.find_controller()

        self.stop_serial_processing_event = Event()
        self.inbound_queue = Queue()
        self.outbound_queue = Queue()

        self.serial_thread = Thread(target=self.__serial_processing, name='serial')
        self.serial_thread.daemon = True

    @staticmethod
    def find_controller() -> str:
        ports = list(filter(lambda port: 'CH340' in port.description, list_ports.comports()))
        match len(ports):
            case 0:
                raise NoSorterException()
            case 1:
                return ports[0].device
            case _:
                comPort = ports[0].device
                _logger.warning(f'Multiple ({len(ports)}) controller ports detected, {comPort} selected')
                return comPort

    def start_processing(self):
        self.stop_serial_processing_event.clear()
        self.serial_thread.start()
        _logger.info('Processing started')

        self.wait_state('Wait')
        self.move()

    def stop_processing(self):
        self.stop_serial_processing_event.set()
        if self.serial_thread.is_alive():
            self.serial_thread.join()
        _logger.info('Processing stopped')

    def change_state(self, state: str, bucket: str = '', confirm: bool = True):
        obj = state + bucket
        _logger.debug(f'Placing command {obj} to inbound queue')
        self.inbound_queue.put(obj)

        if confirm:
            self.wait_confirmation(state)

    def connect(self):
        self.start_processing()

    def wait(self):
        self.change_state('W')
        _logger.info('Wait state set')

    def clean(self):
        self.change_state('C')
        _logger.info('Clean state set')

    def move(self):
        self.change_state('M')
        _logger.info('Move state set')

    def select(self, bucket: chr):
        self.change_state('S', bucket)
        _logger.info(f'Selection state set with bucket {bucket}')

    def recognize(self):
        self.change_state('R')
        _logger.info('Recognition state set')

    def disconnect(self):
        self.stop_processing()

    def __serial_processing(self):
        _logger.info(f'Connecting to Arduino controller on {self.port}')
        arduino = serial.Serial(self.port, 9600, timeout=1)

        while not self.stop_serial_processing_event.is_set():
            response = arduino.readline()
            _logger.debug(f'Response: {response}')

            if (msg := self.__decode_response(response)):
                _logger.debug(f'Message: {msg}')
                self.outbound_queue.put(msg)

            while not self.inbound_queue.empty():
                cmd = self.inbound_queue.get()
                arduino.write(cmd.encode('ascii'))
                _logger.debug(f'Command sent: {cmd}')

    def __decode_response(self, response: bytes) -> dict:
        if not response:
            return None
        
        params = response.decode('ascii').strip().split(':')
        match params[0]:
            case 'S':
                # status message
                return {'message_type': 'S',
                        'state': params[1],
                        'time_on_state': params[2],
                        'vibro_state': params[3],
                        'conveyor_state': params[4]}

            case 'C':
                # confirmation message
                return {'message_type': 'C',
                        'state': params[1]}

            case 'E':
                # error message
                return {'message_type': 'E',
                        'message': params[1]}

            case 'D':
                # debug message, just log it
                _logger.info(f'Debug response: {params}')

            case _:
                # unknown
                _logger.error(f'Unknown response: {response}')
                
        return None

    def wait_response(self, timeout: float = None) -> dict:
        try:
            return self.outbound_queue.get(block=True, timeout=timeout)
        except Empty:
            return None

    def wait_state(self, state: str, timeout: float = None) -> dict:
        try:
            while (msg := self.outbound_queue.get(block=True, timeout=timeout)):
                if msg['message_type'] == 'S' and msg['state'][0] == state[0]:
                    return msg
        except Empty:
            return None

    def wait_confirmation(self, last_command: str, timeout: float = None) -> dict:
        try:
            while (msg := self.outbound_queue.get(block=True, timeout=timeout)):
                if msg['message_type'] == 'C' and msg['state'][0] == last_command[0]:
                    return msg
        except Empty:
            return None
