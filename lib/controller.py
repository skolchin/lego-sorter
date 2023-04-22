# LEGO sorter project
# HW controller prototype
# (c) kol, 2022

import os
import json
import logging
from queue import Queue
from threading import Event, Lock, Thread
import serial
import serial.tools.list_ports
import time
from typing import Tuple, Iterable
from lib.exceptions import NoSorter, NotConnectedToSorter, NotImplemented

from .globals import ROOT_DIR

logger = logging.getLogger(__name__)


class Controller:
    """ HW controller 
        There is only one sorter can be controlled by appliciation, so we use Singleton pattern 
    """
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Controller, cls).__new__(cls)
        return cls.instance

    UNKNOWN: str = '_'

    stop_serial_processing_event = Event()

    serial_thread: Thread = None
    lock = Lock()

    inboundQueue = Queue()
    outboundQueue = Queue()

    def __init__(self):
        # move or remove?
        with open(os.path.join(ROOT_DIR, 'lib', 'bins.json'), 'r') as fp:
            self.bins = json.load(fp)
            if not self.UNKNOWN in self.bins:
                self.bins[self.UNKNOWN] = (-1, -1)

        # init base variables
        self.port = self.find_controller()

        self.serial_thread = Thread(
            target=self.__serial_processing, name='serial')
        self.serial_thread.daemon = True

        print("Controller init")

    @property
    def labels(self) -> Iterable[str]:
        return list(set(self.bins.keys()).difference([self.UNKNOWN]))

    def find_bin(self, label: str) -> Tuple[str]:
        try:
            return tuple(self.bins[label])
        except KeyError:
            return tuple(self.bins[self.UNKNOWN])

    @staticmethod
    def get_ports():
        ports = serial.tools.list_ports.comports()
        return ports

    @staticmethod
    def find_controller():
        portsFound = Controller.get_ports()
        comPort = 'None'
        numConnection = len(portsFound)
        arduino_cnt = 0

        for i in range(0, numConnection):
            port = portsFound[i]
            strPort = str(port)

            if 'Arduino' in strPort or 'CH340' in strPort:
                splitPort = strPort.split(' ')
                comPort = (splitPort[0])
                arduino_cnt += 1

        if comPort == 'None':
            logger.warning("Unable to find lego sorter. Is it connected?")

        if arduino_cnt > 1:
            logger.warning(
                f'There are {arduino_cnt} ports with controller. Selected {comPort}. Please use set_port() to specify correct port')

        return comPort

    def start_processing(self):
        self.stop_processing()
        self.stop_serial_processing_event.clear()
        self.serial_thread.start()

    def stop_processing(self):
        if self.serial_thread is not None and self.serial_thread.is_alive():
            self.stop_serial_processing_event.set()
            self.serial_thread.join()

    def change_state(self, state: chr, bucket: chr = ''):
        if self.serial_thread is None or not self.serial_thread.is_alive():
            logger.error("Controller disconnected")
            raise NotConnectedToSorter

        with self.lock:
            self.inboundQueue.put(state + bucket)

    def connect(self):
        if self.port == 'None':
            logger.error('No port selected')
            raise NoSorter

        self.start_processing()

        logger.info('Controller connected')

    def wait(self):
        logger.info('Wait state called')
        self.change_state('W')

    def run(self):
        if self.serial_thread is None or not self.serial_thread.is_alive():
            logger.error("Controller disconnected")
            return

        logger.info('Run state called')
        self.change_state('M')

    def select(self, bucket: chr):
        if self.serial_thread is None or not self.serial_thread.is_alive():
            logger.error("Controller disconnected")

        logger.info(f'Select state called with bucket {bucket}')
        self.change_state('S', bucket)

    def recognize(self):
        logger.info('Recognostiation state called')
        self.change_state('R')

    def disconnect(self):
        self.stop_processing()

        logger.info('Controller disconnected')

    def __serial_processing(self):
        with self.lock:
            port = self.port

        arduino = serial.Serial(port, 9600)

        while True:
            if self.stop_serial_processing_event.is_set():
                break

            data = arduino.readline()
            self.__process_response_data(str(data))

            with self.lock:
                if not self.inboundQueue.empty():
                    command = self.inboundQueue.get()
                    arduino.write(bytes(command, 'utf-8'))
                    time.sleep(0.05)

    def __process_response_data(self, data):
        data = data[2:-3]
        print(data)
        param_list = data.split(":")

        response = None

        match param_list[0]:
            case "S":
                # status message from arduino controller
                response = {'message_type': 'S',
                            'state': param_list[1],
                            'time_on_state': param_list[2],
                            'vibro_state': param_list[3],
                            'conveyor_state': param_list[4]}

                logger.info(response)
            case "C":
                # confirmation message from arduino controller
                response = {'message_type': 'C',
                            'message': param_list[1]}

                logger.info(response)
            case "E":
                # error message from arduino controller
                response = {'message_type': 'E',
                            'message': param_list[1]}

                logger.error(response)
            case "D":
                # debug message from arduino controller. Just log it
                logger.info(param_list[1])

        with self.lock:
            self.outboundQueue.put(response)

    def get_next_message(self) -> dict:
        response = None
        if not self.outboundQueue.empty():
            response = self.outboundQueue.get()
        return response
