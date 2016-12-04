import gym
from universe import envs, error, spaces, vectorized

def atari_vnc(up=False, down=False, left=False, right=False, z=False):
    return [spaces.KeyEvent.by_name('up', down=up),
            spaces.KeyEvent.by_name('left', down=left),
            spaces.KeyEvent.by_name('right', down=right),
            spaces.KeyEvent.by_name('down', down=down),
            spaces.KeyEvent.by_name('z', down=z)]

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
        elif self.spec.id == 'internet.SlitherIO-v0' or self.spec.id == 'internet.SlitherIOErmiyaEskandaryBot-v0' or self.spec.id == 'internet.SlitherIOEasy-v0':
            self.action_space = spaces.Hardcoded([
                slither_vnc(left=True),
                slither_vnc(right=True),
                slither_vnc(space=True),
                slither_vnc(left=True, space=True),
                slither_vnc(right=True, space=True),
            ])
        elif self.spec.id in ['flashgames.DuskDrive-v0']:
            # TODO: be more systematic
            self.action_space = spaces.Hardcoded([
                racing_vnc(up=True),
                racing_vnc(left=True),
                racing_vnc(right=True),
            ])
        elif self.spec.id in ['flashgames.RedBeard-v0']:
            self.action_space = spaces.Hardcoded([
                platform_vnc(up=True),
                platform_vnc(left=True),
                platform_vnc(right=True),
                platform_vnc(space=True),
            ])

def gym_core_action_space(gym_core_id):
    spec = gym.spec(gym_core_id)

    if spec.id == 'CartPole-v0':
        return spaces.Hardcoded([
            [spaces.KeyEvent.by_name('left', down=True)],
            [spaces.KeyEvent.by_name('left', down=False)],
        ])
    elif spec._entry_point.startswith('gym.envs.atari:'):
        actions = []
        env = spec.make()
        for action in env.unwrapped.get_action_meanings():
            z = 'FIRE' in action
            left = 'LEFT' in action
            right = 'RIGHT' in action
            up = 'UP' in action
            down = 'DOWN' in action
            translated = atari_vnc(up=up, down=down, left=left, right=right, z=z)
            actions.append(translated)
        return spaces.Hardcoded(actions)
    else:
        raise error.Error('Unsupported env type: {}'.format(spec.id))
