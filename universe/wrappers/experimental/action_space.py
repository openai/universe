import logging

import gym
import numpy as np
from universe import spaces
from universe import vectorized
from universe.wrappers.gym_core import gym_core_action_space

logger = logging.getLogger(__name__)

def slither_vnc(space=False, left=False, right=False):
    return [spaces.KeyEvent.by_name('space', down=space),
            spaces.KeyEvent.by_name('left', down=left),
            spaces.KeyEvent.by_name('right', down=right)]

def racing_vnc(up=False, left=False, right=False):
    return [spaces.KeyEvent.by_name('up', down=up),
            spaces.KeyEvent.by_name('left', down=left),
            spaces.KeyEvent.by_name('right', down=right)]

def platform_vnc(up=False, left=False, right=False, space=False):
    return [spaces.KeyEvent.by_name('up', down=up),
            spaces.KeyEvent.by_name('left', down=left),
            spaces.KeyEvent.by_name('right', down=right),
            spaces.KeyEvent.by_name('space', down=space)]


class SafeActionSpace(vectorized.Wrapper):
    """
    Recall that every universe environment receives a list of VNC events as action.
    There exist many environments for which the set of relevant action is much smaller
    and is known.   For example, Atari environments have a modest number of keys,
    so this wrapper, when applied to an Atari environment will reduce its action space.
    Doing so is very convenient for research, since today's RL algorithms rely on random
    exploration, which is hurt by small action spaces.  As our algorithms get better
    and we switch to using the raw VNC commands, this wrapper will become less important.
    """
    def __init__(self, env):
        super(SafeActionSpace, self).__init__(env)

        if self.spec.tags.get('runtime') == 'gym-core':
            self.action_space = gym_core_action_space(self.spec._kwargs['gym_core_id'])
        elif self.spec is None:
            pass
        elif self.spec.id in ['internet.SlitherIO-v0',
                              'internet.SlitherIOErmiyaEskandaryBot-v0',
                              'internet.SlitherIOEasy-v0']:
            self.action_space = spaces.Hardcoded([slither_vnc(left=True),
                                                  slither_vnc(right=True),
                                                  slither_vnc(space=True),
                                                  slither_vnc(left=True, space=True),
                                                  slither_vnc(right=True, space=True)])
        elif self.spec.id in ['flashgames.DuskDrive-v0']:
            # TODO: be more systematic
            self.action_space = spaces.Hardcoded([racing_vnc(up=True),
                                                  racing_vnc(left=True),
                                                  racing_vnc(right=True)])
        elif self.spec.id in ['flashgames.RedBeard-v0']:
            self.action_space = spaces.Hardcoded([platform_vnc(up=True),
                                                  platform_vnc(left=True),
                                                  platform_vnc(right=True),
                                                  platform_vnc(space=True)])


class SoftmaxClickMouse(vectorized.ActionWrapper):
    """
    Creates a Discrete action space of mouse clicks.

    This wrapper divides the active region into cells and creates an action for
    each which clicks in the middle of the cell.
    """
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210), discrete_mouse_step=10, noclick_regions=[]):
        super(SoftmaxClickMouse, self).__init__(env)
        logger.info('Using SoftmaxClickMouse with action_region={}, noclick_regions={}'.format(active_region, noclick_regions))
        xlow, ylow, xhigh, yhigh = active_region
        xs = range(xlow, xhigh, discrete_mouse_step)
        ys = range(ylow, yhigh, discrete_mouse_step)
        self.active_region = active_region
        self.discrete_mouse_step = discrete_mouse_step
        self.noclick_regions = noclick_regions
        self._points = []
        removed = 0
        for x in xs:
            for y in ys:
                xc = min(x+int(discrete_mouse_step/2), xhigh-1) # click to center of a cell
                yc = min(y+int(discrete_mouse_step/2), yhigh-1)
                if any(self.is_contained((xc, yc), r) for r in noclick_regions):
                    removed += 1
                    continue
                self._points.append((xc, yc))
        logger.info('SoftmaxClickMouse noclick regions removed {} of {} actions'.format(removed, removed + len(self._points)))
        self.action_space = gym.spaces.Discrete(len(self._points))

    def _action(self, action_n):
        return [self._discrete_to_action(int(i)) for i in action_n]

    def _discrete_to_action(self, i):
        xc, yc = self._points[i]
        return [
            spaces.PointerEvent(xc, yc, buttonmask=0), # release
            spaces.PointerEvent(xc, yc, buttonmask=1), # click
            spaces.PointerEvent(xc, yc, buttonmask=0), # release
        ]

    def _reverse_action(self, action):
        xlow, ylow, xhigh, yhigh = self.active_region
        try:
            # find first valid mousedown, ignore everything else
            click_event = next(e for e in action if isinstance(e, spaces.PointerEvent) and e.buttonmask == 1)
            index = self._action_to_discrete(click_event)
            if index is None:
                return np.zeros(len(self._points))
            else:
                # return one-hot vector, expected by demo training code
                # FIXME(jgray): move one-hot translation to separate layer
                return np.eye(len(self._points))[index]
        except StopIteration:
            # no valid mousedowns
            return np.zeros(len(self._points))

    def _action_to_discrete(self, event):
        assert isinstance(event, spaces.PointerEvent)
        x, y = event.x, event.y
        step = self.discrete_mouse_step
        xlow, ylow, xhigh, yhigh = self.active_region
        xc = min((int((x - xlow) / step) * step) + xlow + step / 2, xhigh - 1)
        yc = min((int((y - ylow) / step) * step) + ylow + step / 2, yhigh - 1)
        try:
            return self._points.index((xc, yc))
        except ValueError:
            # ignore clicks outside of active region or in noclick regions
            return None

    @classmethod
    def is_contained(cls, point, coords):
        px, py = point
        x, width, y, height = coords
        return x <= px <= x + width and y <= py <= y + height
