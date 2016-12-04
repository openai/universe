import string
from universe import error
from universe.vncdriver import constants

class VNCEvent(object):
    pass

def keycode(key):
    if key in constants.KEYMAP:
        return constants.KEYMAP.get(key)
    elif len(key) == 1:
        return ord(key)
    else:
        raise error.Error('Not sure how to translate to keycode: {!r}'.format(key))

class KeyEvent(VNCEvent):
    _keysym_to_name = {}
    for key, value in constants.KEYMAP.items():
        _keysym_to_name[value] = key
    for c in string.printable:
        _keysym_to_name[ord(c)] = c

    @classmethod
    def build(cls, keys, down=None):
        """Build a key combination, such as:

        ctrl-t
        """
        codes = []
        for key in keys.split('-'):
            key = keycode(key)
            codes.append(key)

        events = []
        if down is None or down:
            for code in codes:
                events.append(cls(code, down=True))

        if down is None or not down:
            for code in reversed(codes):
                events.append(cls(code, down=False))
        return events

    @classmethod
    def by_name(cls, key, down=None):
        return cls(keycode(key), down=down)

    def __init__(self, key, down=True):
        # TODO: validate key
        self.key = key
        self.down = bool(down)

    def compile(self):
        return 'KeyEvent', self.key, self.down

    def __repr__(self):
        if self.down:
            direction = 'down'
        else:
            direction = 'up'
        name = self._keysym_to_name.get(self.key)
        if not name:
            name = '0x{:x}'.format(self.key)
        else:
            name = '{} (0x{:x})'.format(name, self.key)
        return 'KeyEvent<key={} direction={}>'.format(name, direction)

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return (self.key, self.down).__hash__()

    def __eq__(self, other):
        return type(other) == type(self) and \
            other.key == self.key and \
            other.down == self.down

    @property
    def key_name(self):
        """Human readable name"""
        return self._keysym_to_name.get(self.key)

class PointerEvent(VNCEvent):
    def __init__(self, x, y, buttonmask=0):
        self.x = x
        self.y = y
        self.buttonmask = buttonmask

    def compile(self):
        return 'PointerEvent', self.x, self.y, self.buttonmask

    def __repr__(self):
        return 'PointerEvent<x={} y={} buttonmask={}>'.format(self.x, self.y, self.buttonmask)

    def __str__(self):
        return repr(self)
