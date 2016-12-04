import logging
import numpy as np
from universe import pyprofile
import threading
import time

from universe import error
from universe.twisty import reactor
from universe.vncdriver import server_messages
from universe.spaces import vnc_event

logger = logging.getLogger(__name__)

class NumpyScreen(object):
    def __init__(self, width, height):
        self.lock = threading.RLock()

        shape = (height, width, 3)
        self._screens = (np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8))

        self.color_cycle = [0, 1, 2]
        self._width = None
        self._height = None

        self._defer = []

        self.paint_cursor = False
        self._cursor = {
            id(self._screens[0]): {
                'behind': None,
                'details': None,
                'painted': False,
            },
            id(self._screens[1]): {
                'behind': None,
                'details': None,
                'painted': False,
            },
        }

        self._back_updated = True
        self._back_cursor_updated = True

        self.cursor_shape = None
        self.cursor_position = None

        self._has_initial_framebuffer_update = False

    def set_paint_cursor(self, paint_cursor):
        self.paint_cursor = paint_cursor

    def peek(self):
        front_screen, _ = self._screens
        return front_screen

    def flip(self):
        pyprofile.push('vncdriver.numpy_screen.flip_bitmap')
        with self.lock:
            if self._back_updated:
                updates = self._defer

                # Flip screens
                front_screen, back_screen = self._screens
                self._screens = back_screen, front_screen

                # Mark ourselves as pending application of updates
                self._back_updated = False

                # This can be called asynchronously if desired, but it means
                # less reliably smooth playback.
                #
                # reactor.callFromThread(self.update_back)
                self.update_back()
            else:
                updates = []
            result = self.peek(), {'vnc_session.framebuffer_updates': updates}
        pyprofile.pop()
        return result

    def apply_action(self, action):
        if isinstance(action, vnc_event.PointerEvent):
            with self.lock:
                self.cursor_position = (action.x, action.y)

                # If not self._back_updated, we're not actually up to
                # date, so any pixels we cached would be wrong. When
                # back updates, it'll automatically render the cursor.
                if self._back_updated and self.paint_cursor:
                    self._unpaint_cursor()
                    self._paint_cursor()

    def apply(self, framebuffer_update):
        with self.lock:
            self._has_initial_framebuffer_update = True

            # Pop any pending updates
            self._update_back()
            self._apply(framebuffer_update)
            self._defer.append(framebuffer_update)


    def _apply(self, framebuffer_update):
        if self.paint_cursor:
            self._unpaint_cursor()
        for rect in framebuffer_update.rectangles:
            if isinstance(rect.encoding,
                          (server_messages.RAWEncoding, server_messages.ZRLEEncoding, server_messages.ZlibEncoding)):
                self._update_rectangle(rect.x, rect.y, rect.width, rect.height, rect.encoding.data)
            elif isinstance(rect.encoding, server_messages.PseudoCursorEncoding):
                self._update_cursor_shape(rect.x, rect.y, rect.width, rect.height, rect.encoding.image, rect.encoding.mask)
            else:
                raise error.Error('Unrecognized encoding: {}'.format(rect.encoding))
        if self.paint_cursor:
            self._paint_cursor()

    def update_back(self):
        with self.lock:
            self._update_back()

    def _update_back(self):
        if self._back_updated:
            return
        self._back_updated = True

        for framebuffer_update in self._defer:
            self._apply(framebuffer_update)

        if len(self._defer) == 0 and self.paint_cursor:
            self._unpaint_cursor()
            self._paint_cursor()

        self._defer = []

    def _update_rectangle(self, x, y, width, height, data):
        _, back_screen = self._screens
        back_screen[y:y+height, x:x+width, self.color_cycle] = data

    def _update_cursor_shape(self, hotx, hoty, width, height, image, mask):
        # hotx, hoty are the hotspot within the cursor
        self.cursor_shape = (hotx, hoty, width, height, image, mask)

    def _paint_cursor(self):

        # use our knowledge of the x, y cursor position plus the
        # cursor shape to paint the cursor
        if self.cursor_position is None:
            return
        elif not self._has_initial_framebuffer_update:
            return
        elif self.cursor_shape is None:
            return

        self._back_cursor_updated = True

        _, back_screen = self._screens
        cursor = self._cursor[id(back_screen)]

        assert not cursor['painted']
        cursor['painted'] = True

        hotx, hoty, width, height, image, mask = self.cursor_shape
        x, y = self.cursor_position

        # Save old data
        cursor['details'] = (x, y, width, height)
        cursor['behind'] = back_screen[y:y+height, x:x+width, :].copy()

        # Paint the cursor
        total_h, total_w, _ = back_screen.shape

        cutoff_h = min(total_h - y, height)
        cutoff_w = min(total_w - x, width)
        image = image[:cutoff_h, :cutoff_w]
        mask = mask[:cutoff_h, :cutoff_w, np.newaxis]

        back_screen[y:y+height, x:x+width, self.color_cycle] = (1 - mask)*back_screen[y:y+height, x:x+width, self.color_cycle] + mask*image

    def _unpaint_cursor(self):

        _, back_screen = self._screens
        cursor = self._cursor[id(back_screen)]

        if cursor['behind'] is not None:
            assert cursor['painted']
            x, y, width, height = cursor['details']
            back_screen[y:y+height, x:x+width, :] = cursor['behind']
        cursor['painted'] = False


    # def _copy_rectangle(self, screen, src_x, src_y, x, y, width, height):
    #     update = np.frombuffer(data, dtype=np.uint8)
    #     update = update.reshape([height, width, 4])
    #     update = update[:, :, self._color_cycle]  # Ignore X channel


    #     screen[y:y+height, x:x+width] = screen[src_y:src_y+height, src_x:src_x+width]

    # def _fill_rectangle(self, screen, x, y, width, height, color):
    #     update = np.frombuffer(color, dtype=np.uint8)
    #     update = update[self._color_cycle]
    #     screen[y:y+height, x:x+width] = update

    # def begin(self, number_of_rectangles):
    #     self.lock.acquire()
    #     # This may already have been called via
    #     # reactor.callFromThread. It's safe to be called multiple times.
    #     self._update_back()

    # def commit(self):
    #     self.lock.release()

    # def back_bitmap(self):
    #     _, back_screen = self._screens
    #     return back_screen

    # def screen_synced(self):
    #     # TODO: lock?
    #     return self._screens is not None
