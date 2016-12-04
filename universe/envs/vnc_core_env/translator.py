from universe import spaces
from universe.envs.vnc_core_env import key
import logging

logger = logging.getLogger(__name__)


class AtariKeyState(object):
    """
    Converts from VNCEvents to an Atari-v0 action index

    Since spaces.KeyEvent only give you a diff of a keyboard, we need to persist the total state of the keyboard to
    convert from VNCEvents to an action index
    """
    def __init__(self, env):
        self._translator = AtariTranslator(env)
        self._down_keysyms = set()  # Assumes that your env starts with no keys pressed down

    def apply_vnc_actions(self, vnc_actions):
        """
        Play a list of vnc_actions forward over the current keysyms state

        NOTE: Since we are squashing a set of diffs into a single keyboard state, some information may be lost.
        For example if the Z key is down, then we receive [(Z-up), (Z-down)], the output will not reflect any change in Z
        You can make each frame shorter to offset this effect.
        """
        for event in vnc_actions:
            if isinstance(event, spaces.KeyEvent):
                if event.down:
                    self._down_keysyms.add(event.key)
                else:
                    self._down_keysyms.discard(event.key)

        logger.debug("AtariKeyState._down_keysyms: {}".format(self._down_keysyms))

    def to_keysyms(self):
        """Returns the current state as keysyms"""
        return list(self._down_keysyms)

    def to_index(self):
        """Returns the current state as an index"""
        return self._translator.keysyms_to_index(self.to_keysyms())


class AtariTranslator(object):
    """Translates Atari actions to and from various formats"""
    _all_keysyms = [key.UP, key.DOWN, key.LEFT, key.RIGHT, key.Z]

    def __init__(self, env):
        # e.g. {0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT', 4: 'RIGHTFIRE', 5: 'LEFTFIRE'}
        self._index_to_name_ = {}
        # e.g. {'RIGHT': 2, 'FIRE': 1, 'RIGHTFIRE': 4, 'LEFTFIRE': 5, 'NOOP': 0, 'LEFT': 3}
        self._name_to_index_ = {}

        for i, meaning in enumerate(env.unwrapped.get_action_meanings()):
            self._name_to_index_[meaning] = i
            self._index_to_name_[i] = meaning

    def keysyms_to_vnc_actions(self, keysyms):
        actions = []
        keysyms = set(keysyms)
        for keysym in self._all_keysyms:
            down = keysym in keysyms
            actions.append(spaces.KeyEvent(keysym, down=down))
        return actions

    def keysyms_to_index(self, keysyms):
        name = self._keysyms_to_name(keysyms)
        return self._name_to_index(name)

    def index_to_keysyms(self, i):
        name = self._index_to_name(i)
        keysyms = []
        if 'UP' in name:
            keysyms.append(key.UP)
        if 'DOWN' in name:
            keysyms.append(key.DOWN)
        if 'LEFT' in name:
            keysyms.append(key.LEFT)
        if 'RIGHT' in name:
            keysyms.append(key.RIGHT)
        if 'FIRE' in name:
            keysyms.append(key.Z)
        return keysyms

    def _name_to_index(self, name):
        return self._name_to_index_.get(name, 0)

    def _index_to_name(self, i):
        return self._index_to_name_[i]

    def _keysyms_to_name(self, keysyms):
        keys = ''
        if key.UP in keysyms:
            keys += 'UP'
        if key.DOWN in keysyms:
            keys += 'DOWN'
        if key.LEFT in keysyms:
            keys += 'LEFT'
        if key.RIGHT in keysyms:
            keys += 'RIGHT'
        if key.Z in keysyms:
            keys += 'FIRE'
        return keys

class CartPoleTranslator(object):
    def __init__(self, env):
        pass

    def keysyms_to_vnc_actions(self, keysyms):
        down = key.LEFT in keysyms
        return [spaces.KeyEvent(key.LEFT, down=down)]

    def keysyms_to_index(self, keys):
        if key.LEFT in keys:
            return 0
        else:
            return 1

    def index_to_keysyms(self, i):
        if i == 0:
            return [key.LEFT]
        else:
            return []
