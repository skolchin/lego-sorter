# LEGO sorter project
# Debug and status info class
# (c) lego-sorter team, 2022-2023

import cv2
import numpy as np
import logging
from typing import Iterable

_logger = logging.getLogger('lego-tracker')

class StatusInfo:
    """ Status information display helper class.

    Can show some textual info in a separate window or draw it on an image.
    Texts are displayed in a top-left corner, one line under another. 
    Text color is set according to importance provied (either red for important messages or green for normal one).
    Please note that images are assumed to be in OpenCV format (BGR), not in standard RGB.
    Text font size is currently fixed, will work this out if needed. 
    Allows status "scrolling" - no more than given number of text lines is displayed simultaneously.

    Arguments:
        title:  window title, used if displayed in a separate window
        size: window size, used if displayed in a separate window
        max_len: maximum number of status lines to keep and show
        max_width: maximum length of a line (not used now)

    Examples:
        >>> # instantly draws given text on image
        >>> StatusInfo().assign_and_apply(image, 'Some text', important=True)
        >>> cv2.imshow('image', image)
        >>>
        >>> # Add several statuses, but display only last two
        >>> st = StatusInfo(max_len=2)
        >>> for status in ['one', 'two', 'three']:
        >>>   st.append(status)
        >>> st.apply(image)
        >>> cv2.imshow('image', image)
        >>>
        >>> # show status in a separate window
        >>> st = StatusInfo()
        >>> st.append('Current status')
        >>> st.show()
    """
    def __init__(self, title: str = 'status', size: tuple = (240, 320), max_len: int = 0, max_width: int = 70):
        self.title = title
        self.size = size
        self.max_len = max_len
        self.max_width = max_width
        self.__status = []
        self.__visible = False

    @property
    def visible(self):
        """ Status window visibility """
        return self.__visible

    @visible.setter
    def visible(self, value: bool):
        """ Status window visibility """
        if value:
            self.show()
        else:
            self.hide()

    def show(self):
        """ Show status window """
        img = self._make_image()
        cv2.imshow(self.title, img)
        self.__visible = True

    def hide(self):
        """ Hide status window """
        if self.visible:
            try:
                cv2.destroyWindow(self.title)
                self.__visible = False
            except:
                pass

    def insert(self, at: int, status: str, important: bool = False):
        """ Insert status message at given position in the list """
        self.__status.insert(at, (status, important))
        _logger.info(status)
        if self.max_len:
            while len(self.__status) > self.max_len:
                del self.__status[-1]
        if self.visible:
            self.show()

    def append(self, status: str, important: bool = False):
        """ Append status message to the end of the list """
        self.__status.append((status, important))
        _logger.info(status)
        if self.max_len:
            while len(self.__status) > self.max_len:
                del self.__status[0]
        if self.visible:
            self.show()

    def assign(self, statuses: Iterable[str], important: bool = False):
        """ Completelly replace list of statuses """
        self.__status = []
        for status in statuses:
            self.__status.append((status, important))
        if self.visible:
            self.show()

    def clear(self):
        """ Clear list of statuses """
        self.__status.clear()
        if self.visible:
            self.show()

    def __delitem__(self, at: int):
        del self.__status[at]
        if self.visible:
            self.show()

    def _make_image(self, img: np.ndarray = None) -> np.ndarray:
        """ Internal - apply statuses to existing image or construct new one """
        if img is None:
            img = np.full(list(self.size) + [3], (255,255,255), np.uint8)
        x, y = 10, 14
        for (status, important) in self.__status:
            clr = (0,0,255) if important else (0,255,0)
            cv2.putText(img, status, (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, .6, color=clr)
            y += 16
        return img

    def apply(self, img: np.ndarray) -> np.ndarray:
        """ Apply collected statuses to the image """
        return self._make_image(img)

    def assign_and_apply(self, img: np.ndarray, status: str, important: bool = False):
        """ Replaces current statuses list with given status and apply it to the image """
        self.assign([status], important)
        return self.apply(img)
