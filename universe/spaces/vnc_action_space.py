import gym
import string

from gym.spaces import prng

from universe.vncdriver import constants
from universe.spaces import vnc_event

class VNCActionSpace(gym.Space):
    """The space of VNC actions.

    You can submit a list of KeyEvents or PointerEvents. KeyEvents
    correspond to pressing or releasing a key. PointerEvents correspond
    to moving to a specific pixel, and setting the mouse buttons to some state
    (buttonmask is a bitmap corresponding to which buttons are down).

    Note that key releases work differently from click releases: keys
    are stateful and must be explicitly released, while the state of
    the mouse buttons is provided at each timestep, so you have to
    explicitly keep the mouse down.

    Attributes:
        keys (list<KeyEvent>): The allowed key presses
        buttonmasks (list<int>): The allowed buttonmasks (i.e. mouse presses)
        screen_shape (int, int): The X and Y dimensions of the screen
    """

    def __init__(self, keys=None, buttonmasks=None, screen_shape=(1024, 728)):
        self.keys = []
        if keys is None:
            keys = [c for c in string.printable] + list(constants.KEYMAP.keys())
        for key in (keys or []):
            down = vnc_event.KeyEvent.by_name(key, down=True)
            up = vnc_event.KeyEvent.by_name(key, down=False)
            self.keys.append(down)
            self.keys.append(up)
        self._key_set = set(self.keys)

        self.screen_shape = screen_shape
        if self.screen_shape is not None:
            self.buttonmasks = []
            if buttonmasks is None:
                buttonmasks = range(256)
            for buttonmask in buttonmasks:
                self.buttonmasks.append(buttonmask)
            self._buttonmask_set = set(self.buttonmasks)

    def contains(self, action):
        if not isinstance(action, list):
            return False

        for a in action:
            if isinstance(a, vnc_event.KeyEvent):
                if a not in self._key_set:
                    return False
            elif isinstance(a, vnc_event.PointerEvent):
                if self.screen_shape is None:
                    return False

                if a.x < 0 or a.x > self.screen_shape[0]:
                    return False
                elif a.y < 0 or a.y > self.screen_shape[1]:
                    return False
                elif a.buttonmask not in self._buttonmask_set:
                    return False

        return True

    def sample(self):
        # Both key and pointer allowed
        if self.screen_shape is not None:
            event_type = prng.np_random.randint(2)
        else:
            event_type = 0

        if event_type == 0:
            # Let's press a key
            key = prng.np_random.choice(self.keys)
            event = [key]
        else:
            x = prng.np_random.randint(self.screen_shape[0])
            y = prng.np_random.randint(self.screen_shape[1])
            buttonmask = prng.np_random.choice(self.buttonmasks)

            event = [vnc_event.PointerEvent(x, y, buttonmask)]
        return event
