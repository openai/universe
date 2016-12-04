import logging
import time
import threading

from universe.vncdriver.screen import numpy_screen

logger = logging.getLogger(__name__)

class ScreenBuffer(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.uncommitted = []
        self.updates = []

    def apply_format(self, attrs):
        self._push({
            'type': 'apply_format',
            'attrs': attrs,
        })

    def update_rectangle(self, x, y, width, height, data):
        self._push({
            'type': 'update_rectangle',
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'data': data,
        })

    def copy_rectangle(self, src_x, src_y, x, y, width, height):
        self._push({
            'type': 'copy_rectangle',
            'src_x': src_x,
            'src_y': src_y,
            'x': x,
            'y': y,
            'width': width,
            'height': height,
        })

    def fill_rectangle(self, x, y, width, height, color):
        self._push({
            'type': 'fill_rectangle',
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'color': color,
        })

    def framebuffer_update_finish(self):
        with self.lock:
            self.updates += self.uncommitted
            self.uncommitted = []

    def _push(self, update):
        """Always call from single thread."""
        self.uncommitted.append(update)

    def pop(self):
        with self.lock:
            if self.updates:
                updates = self.updates
                self.updates = []
                return updates
            else:
                return None

    def peek(self):
        with self.lock:
            if self.updates:
                return self.updates
            else:
                return None
