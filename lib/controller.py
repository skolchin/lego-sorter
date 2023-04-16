# LEGO sorter project
# HW controller prototype
# (c) kol, 2022

import os
import json
import logging
import eventlet
import serial
import serial.tools.list_ports
import time
from typing import Tuple, Iterable
from lib.camera import Camera
from lib.exceptions import NoSorter, NotConnectedToSorter, NotImplemented

from .globals import ROOT_DIR

logger = logging.getLogger(__name__)


class Controller:
    """ HW controller 
        There is only one sorter can be controlled by appliciation, so we use Singleton pattern 
        We use eventlet.sleep() to release process for eventlet async scheduler. 
    """
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Controller, cls).__new__(cls)
        return cls.instance

    UNKNOWN: str = '_'
    connected = False
    active = False

    status_callback = NotImplemented.raise_this
    error_fallback = NotImplemented.raise_this
    confirmation_callback = NotImplemented.raise_this

    def __init__(self):
        # move or remove?
        with open(os.path.join(ROOT_DIR, 'lib', 'bins.json'), 'r') as fp:
            self.bins = json.load(fp)
            if not self.UNKNOWN in self.bins:
                self.bins[self.UNKNOWN] = (-1, -1)

        # init base variables
        self.camera = Camera(self.recognize, self.process_object)
        self.port = self.find_controller()

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

    def get_camera(self):
        return self.camera

    def find_controller(self):
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

    def change_state(self, state: chr, bucket: chr = ''):
        if not self.active:
            logger.error("Controller disconnected")
            raise NotConnectedToSorter

        command = state + bucket

        self.arduino.write(bytes(command, 'utf-8'))
        # blocking sleep
        time.sleep(0.05)

    def activate_camera(self):
        self.camera.stop()
        eventlet.sleep(0.1)
        self.camera.gen_frames()

    def connect(self, status_callback, confirmation_callback, error_fallback):
        if self.port == 'None':
            logger.error('No port selected')
            raise NoSorter

        self.arduino = serial.Serial(self.port, 9600)
        eventlet.sleep(2)
        self.active = True

        self.status_callback = status_callback
        self.confirmation_callback = confirmation_callback
        self.error_fallback = error_fallback

        logger.info('Controller connected')
        self.process()

    def wait(self):
        logger.info('Wait state called')
        self.change_state('W')

    def run(self):
        if not self.active:
            logger.error("Controller disconnected")

        logger.info('Run state called')
        self.camera.capture_background()
        self.change_state('M')

    def select(self, bucket: chr):
        if not self.active:
            logger.error("Controller disconnected")

        logger.info(f'Select state called with bucket {bucket}')
        self.change_state('S', bucket)

    def recognize(self):
        logger.info('Recognostiation state called')
        self.change_state('R')

    def disconnect(self):
        self.active = False

        self.camera.stop()

        logger.info('Controller disconnected')

    def process(self):
        while self.active:
            eventlet.sleep(1)
            data = self.arduino.readline()
            self.process_controller_data(str(data))

    def process_object(self):
        logger.info(f'Recognize stopped object here')
        self.select("D")  # TODO: replace with bucket selection

    def process_controller_data(self, data):
        data = data[2:-3]
        print(data)
        param_list = data.split(":")

        match param_list[0]:
            case "S":
                status = {'state': param_list[1],
                          'time_on_state': param_list[2],
                          'vibro_state': param_list[3],
                          'conveyor_state': param_list[4]}

                self.status_callback(status)
            case "C":
                # confirmation message from arduino controller
                confirmation_message = param_list[1]
                logger.info(confirmation_message)

                self.confirmation_callback(confirmation_message)
            case "E":
                # error message from arduino controller
                error_message = param_list[1]
                logger.error(error_message)

                self.error_fallback(error_message)
            case "D":
                # debug message from arduino controller. Just log it
                logger.info(param_list[1])
