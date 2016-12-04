import logging
import numpy as np
import os
from universe import pyprofile
import sys

from universe import error
from universe.vncdriver import server_messages

logger = logging.getLogger(__name__)

class PygletScreen(object):
    def __init__(self, bitmap=None):
        self._window = None
        self._is_updated = False
        self._height, self._width, _ = bitmap.shape
        self._initialize()
        self.update_rectangle(0, 0, self._width, self._height, bitmap)

    def flip(self):
        if not self._is_updated:
            return
        self._is_updated = False

        self._window.clear()
        self._window.switch_to()
        self._window.dispatch_events()
        self.texture.blit(0, 0)
        self._window.flip()

    def _initialize(self):
        if not os.environ.get('DISPLAY') and sys.platform.startswith('linux'):
            raise error.Error("Cannot render with mode='human' with no DISPLAY variable set.")

        import pyglet
        self._window = pyglet.window.Window(width=self._width, height=self._height, visible=True)
        self._window.dispatch_events()
        self.texture = pyglet.image.Texture.create(width=self._width, height=self._height)

    def update_rectangle(self, x, y, width, height, data):
        bytes = data.tobytes()
        pyprofile.incr('vncdriver.pyglet_screen.blit')
        pyprofile.incr('vncdriver.pyglet_screen.blit.bytes', len(bytes), unit=pyprofile.BYTES)
        import pyglet
        image = pyglet.image.ImageData(width, height, 'RGB', bytes, pitch=width * -3)
        self.texture.blit_into(image, x, self._height-height-y, 0)
        self._is_updated = True

    def apply(self, framebuffer_update):
        pyprofile.push('vncdriver.pyglet_screen.apply')
        for rect in framebuffer_update.rectangles:
            if isinstance(rect.encoding,
                          (server_messages.RAWEncoding, server_messages.ZRLEEncoding, server_messages.ZlibEncoding)):
                self.update_rectangle(rect.x, rect.y, rect.width, rect.height, rect.encoding.data)
            else:
                raise error.Error('Unrecognized encoding: {}'.format(rect.encoding))
        pyprofile.pop()



    # # TODO: we don't seem to be able to have multiple independent
    # # windows at once
    # def update_rectangle(self, x, y, width, height, data):
    #     self._update_rgbarray(x, y, width, height, update)


    # def copy_rectangle(self, src_x, src_y, x, y, width, height):
    #     assert self._window
    #     rectangle = self.texture.get_region(src_x, self._height-height-src_y, width, height)
    #     self.texture.blit_into(rectangle.get_image_data(), x, self._height-height-y, 0)

    # def fill_rectangle(self, x, y, width, height, color):
    #     import pyglet
    #     # While this technically works, it's super slow
    #     update = np.frombuffer(color, dtype=np.uint8)
    #     r, g, b = update[self._color_cycle]
    #     image_pattern = pyglet.image.SolidColorImagePattern(color=(r, g, b, 0))
    #     image = image_pattern.create_image(width, height)
    #     self.texture.blit_into(image, x, self._height-height-y, 0)

    # def commit(self):
    #     self._window.clear()
    #     self._window.switch_to()
    #     self.texture.blit(0, 0)

    #     self._is_updated = True
