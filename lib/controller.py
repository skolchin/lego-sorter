# LEGO sorter project
# HW controller prototype
# (c) lego-sorter team, 2022-2023

import logging
import serial
import time
from serial.tools import list_ports
from queue import Queue
from threading import Event, Lock, Thread
from typing import Mapping

from .exceptions import NoSorter, NotConnectedToSorter

_logger = logging.getLogger('lego-sorter')

class Controller:
    """ HW controller 
        There is only one sorter can be controlled by appliciation, so we use Singleton pattern 
    """
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Controller, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.port = self.find_controller()

        self.stop_serial_processing_event = Event()
        self.lock = Lock()
        self.inboundQueue = Queue()
        self.outboundQueue = Queue()
        self.current_state: chr = ''

        self.serial_thread = Thread(
            target=self.__serial_processing, name='serial')
        self.serial_thread.daemon = True

    @staticmethod
    def get_ports() -> Mapping:
        ports = {}
        for port in list_ports.comports():
            port = str(port)
            splitPort = port.split(' ')
            ports[splitPort[0]] = port[port.find('USB'):]
        return ports

    @staticmethod
    def find_controller() -> str:
        comPort = None
        arduino_cnt = 0

        for port in Controller.get_ports().values():
            if 'Arduino' in port or 'CH340' in port:
                comPort = port.split(' ')[0]
                arduino_cnt += 1

        if comPort is None:
            _logger.warning("Unable to find lego sorter. Is it connected?")

        if arduino_cnt > 1:
            _logger.warning(
                f'There are {arduino_cnt} ports with controller. Selected {comPort}. Please use set_port() to specify correct port')

        return comPort

    def start_processing(self):
        self.stop_processing()
        self.stop_serial_processing_event.clear()
        self.serial_thread.start()
        _logger.info("Processing started")

    def stop_processing(self):
        if self.serial_thread is not None and self.serial_thread.is_alive():
            self.stop_serial_processing_event.set()
            self.serial_thread.join()
        _logger.info("Processing stopped")

    def change_state(self, state: chr, bucket: chr = ''):
        if self.serial_thread is None or not self.serial_thread.is_alive():
            _logger.error("Controller disconnected")
            raise NotConnectedToSorter

        with self.lock:
            obj = state + bucket
            _logger.info(f"Inbound queue put {obj}")
            self.inboundQueue.put(obj)

    def connect(self):
        if not self.port:
            _logger.error('No port selected')
            raise NoSorter

        self.start_processing()
        _logger.info('Controller connected')

    def wait(self):
        _logger.info('Wait state set')
        self.change_state('W')

    def clean(self):
        _logger.info('Clean state set')
        self.change_state('C')

    def run(self):
        if self.serial_thread is None or not self.serial_thread.is_alive():
            _logger.error("Controller disconnected")
            return

        _logger.info('Run state set')
        self.change_state('M')

    def select(self, bucket: chr):
        if self.serial_thread is None or not self.serial_thread.is_alive():
            _logger.error("Controller disconnected")

        _logger.info(f'Selection state set with bucket {bucket}')
        self.change_state('S', bucket)

    def recognize(self):
        _logger.info('Recognition state set')
        self.change_state('R')

    def disconnect(self):
        self.stop_processing()

        _logger.info('Controller disconnected')

    def __serial_processing(self):
        with self.lock:
            port = self.port

        arduino = serial.Serial(port, 9600)

        confirmation_wait = False
        current_command = None

        while True:
            if self.stop_serial_processing_event.is_set():
                break

            data = arduino.readline()
            params = self.__prepare_params(data)

            match params[0]:
                case "C":
                    # confirmation
                    if current_command in params[1]:
                        _logger.info(f"Command {current_command} confirmed")
                        confirmation_wait = False
                    else:
                        _logger.error(
                            f"Command {current_command} unconfirmed. Confirmation for {params}")
                case "E":
                    # error
                    confirmation_wait = False

            self.__process_response_data(params)

            with self.lock:
                if not self.inboundQueue.empty() and not confirmation_wait:
                    current_command = self.inboundQueue.get()
                    _logger.info(f"Inbound queue get command {current_command}")

                    if current_command != self.current_state:
                        arduino.write(bytes(current_command, 'utf-8'))
                        time.sleep(0.1)
                        _logger.info(
                            f"Command {current_command} sent; Wait for confirmation")
                        confirmation_wait = True
                    else:
                        _logger.info(
                            f"Already in state {self.current_state}. Command {current_command} ignored. ")

    def __prepare_params(self, data):
        sdata = str(data)[2:-3]
        return sdata.split(":")

    def __process_response_data(self, params):
        response = None

        match params[0]:
            case "S":
                # status message from arduino controller
                self.current_state = params[1]
                try:
                    response = {'message_type': 'S',
                                'state': params[1],
                                'time_on_state': params[2],
                                'vibro_state': params[3],
                                'conveyor_state': params[4]}

                    _logger.info(response)
                except Exception as e:
                    response = {'message_type': 'E',
                                'message': f"Exception while status decode: {str(e)}; status is: {params}"}

                    _logger.error(response)
            case "C":
                # confirmation message from arduino controller
                response = {'message_type': 'C',
                            'message': params[1]}

                _logger.info(response)
            case "E":
                # error message from arduino controller
                response = {'message_type': 'E',
                            'message': params[1]}

                _logger.error(response)
            case "D":
                # debug message from arduino controller. Just log it
                _logger.info(params[1])

        with self.lock:
            self.outboundQueue.put(response)

    def get_next_message(self) -> dict:
        response = None
        if not self.outboundQueue.empty():
            response = self.outboundQueue.get()
        return response
