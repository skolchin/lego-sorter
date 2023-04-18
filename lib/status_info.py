# LEGO sorter project
# Debug and status info class
# (c) kol, 2022

import cv2
import numpy as np
import img_utils22 as imu
import logging
from typing import Iterable

_logger = logging.getLogger('lego-tracker')

class StatusInfo:
    def __init__(self, title: str = 'status', size: tuple = (240, 320), max_len: int = 0, max_width: int = 70):
        self.title = title
        self.size = size
        self.max_len = max_len
        self.max_width = max_width
        self.__status = []
        self.__visible = False

    @property
    def visible(self):
        return self.__visible

    @visible.setter
    def visible(self, value: bool):
        if value:
            self.show()
        else:
            self.hide()

    def show(self):
        img = self._make_image()
        cv2.imshow(self.title, img)
        self.__visible = True

    def hide(self):
        if self.visible:
            try:
                cv2.destroyWindow(self.title)
                self.__visible = False
            except:
                pass

    def insert(self, at: int, status: str, important: bool = False):
        self.__status.insert(at, (status, important))
        _logger.info(status)
        if self.max_len:
            while len(self.__status) > self.max_len:
                del self.__status[-1]
        if self.visible:
            self.show()

    def append(self, status: str, important: bool = False):
        self.__status.append((status, important))
        _logger.info(status)
        if self.max_len:
            while len(self.__status) > self.max_len:
                del self.__status[0]
        if self.visible:
            self.show()

    def assign(self, statuses: Iterable[str], important: bool = False):
        self.__status = []
        for status in statuses:
            self.__status.append((status, important))
        if self.visible:
            self.show()

    def clear(self):
        self.__status.clear()
        if self.visible:
            self.show()

    def __delitem__(self, at: int):
        del self.__status[at]
        if self.visible:
            self.show()

    def _make_image(self, img: np.ndarray = None) -> np.ndarray:
        if img is None:
            img = np.full(list(self.size) + [3], imu.COLOR_WHITE, np.uint8)
        x, y = 10, 14
        for (status, important) in self.__status:
            clr = imu.COLOR_RED if important else imu.COLOR_GREEN
            cv2.putText(img, status, (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, .6, color=clr)
            y += 16
        return img

    def apply(self, img: np.ndarray) -> np.ndarray:
        return self._make_image(img)

    def assign_and_apply(self, img: np.ndarray, status: str, important: bool = False):
        self.assign([status], important)
        return self.apply(img)
