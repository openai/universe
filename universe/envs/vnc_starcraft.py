import string

from universe import spaces
from universe.spaces import vnc_event, VNCActionSpace
from universe.spaces.vnc_event import KeyEvent, PointerEvent
from universe.envs import vnc_env
from universe.vncdriver import constants
import logging

logger = logging.getLogger()

SCREEN_DIM = (640, 480)

class StarCraftEnv(vnc_env.VNCEnv):
    def __init__(self):
        super(StarCraftEnv, self).__init__()
        self.action_space = VNCActionSpace(
            keys=['f2',  # Map positions
                  'f3',  # Map positions
                  'f4',  # Map positions
                  'spacebar',
                  'left',
                  'up',
                  'right',
                  'down'],
            screen_shape=SCREEN_DIM
        )
        self.safe_action_space = self.action_space

    # def _step(self, action_n):
    #     return super(StarCraftEnv, self)._step(
    #         (StarCraftEventFilter.filter(a) for a in action_n))


# class StarCraftEventFilter(object):
#     """
#     We only allow keyboard inputs used by StarCraft:
#     http://gamingweapons.com/image/steelseries/zboard-starcraft2-keyset/steelseries_zboard_starcraft2_keyset_02.jpg
#     """
#     _x_offset = 5  # Centered
#     _y_offset = 30  # Remove the chrome

#     @classmethod
#     def _safe_pointer_event(cls, event):
#         """Returns true if the click is in a place that will not break out of the box"""
#         height = SCREEN_DIM[0]
#         width = SCREEN_DIM[1]
#         margin = 5  # Never allow clicking within 5 pixels of the edge of the screen

#         unsafe_locations = [
#             (event.y < cls._y_offset + margin),  # At the top, where menu chrome is
#             (event.y > height + cls._y_offset - margin),  # Too far down
#             (event.x < cls._x_offset + margin),  # Too far left
#             (event.x > width + cls._x_offset - margin),  # Too far right
#             (410 < event.x < 510) and (370 < event.y < 450),  # Where the menu button is
#         ]
#         unsafe = any(unsafe_locations)
#         if unsafe:
#             logger.warning('skipping unsafe pointer event')
#         return not unsafe

#     @classmethod
#     def safe_event(cls, event):
#         if isinstance(event, PointerEvent):
#             return cls._safe_pointer_event(event)

#     @classmethod
#     def filter(cls, events):
#         return filter(cls.safe_event, events)
