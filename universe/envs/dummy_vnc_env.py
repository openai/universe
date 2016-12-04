import logging
import numpy as np

from gym.utils import reraise

from universe import error, rewarder, spaces, utils, vectorized
from universe.envs import diagnostics
from universe.remotes import healthcheck
from universe.runtimes import registration

class DummyVNCEnv(vectorized.Env):
    """
    A simple env for unit testing that does nothing, but looks like a VNC env.
    It accepts any actions, and returns black screens.
    It also returns the actions in the observation, so you can test that action wrappers are producing the right answers
    For example, to test that YourActionWrapper converts example_input_action to example_output_action:

    >>> dummy_env = gym.make('test.DummyVNCEnv-v0')
    >>> e = YourActionWrapper(dummy_env)
    >>> e = universe.wrappers.Unvectorize(e)
    >>> observation, reward, done, info = e.step(example_input_action)
    >>> assert observation['action'] == example_output_action

    """
    metadata = {
        'render.modes': ['human'], # we wrap with a Render which can render to rgb_array
        'semantics.async': True,
        'semantics.autoreset': True,
        'video.frames_per_second' : 60,
        'runtime.vectorized': True,
    }

    def __init__(self):
        self._started = False

        self.observation_space = spaces.VNCObservationSpace()
        self.action_space = spaces.VNCActionSpace()

    def _configure(self, remotes=None,
                   client_id=None,
                   start_timeout=None, docker_image=None,
                   ignore_clock_skew=False, disable_action_probes=False,
                   vnc_driver=None, vnc_kwargs={},
                   replace_on_crash=False, allocate_sync=True,
                   observer=False,
                   _n=3,
    ):
        super(DummyVNCEnv, self)._configure()
        self.n = _n
        self._reward_buffers = [rewarder.RewardBuffer('dummy:{}'.format(i)) for i in range(self.n)]
        self._started = True

    def _reset(self):
        return [None] * self.n

    def _step(self, action_n):
        assert self.n == len(action_n), "Expected {} actions but received {}: {}".format(self.n, len(action_n), action_n)

        observation_n = [{
            'visual': np.zeros((1024, 768, 3), dtype=np.uint8),
            'text': [],
            'action': action_n[i]
        } for i in range(self.n)]

        reward_n = []
        done_n = []
        info_n = []
        for reward_buffer in self._reward_buffers:
            reward, done, info = reward_buffer.pop()
            reward_n.append(reward)
            done_n.append(done)
            info_n.append(info)
        return observation_n, reward_n, done_n, {'n': info_n}

    def __str__(self):
        return 'DummyVNCEnv'
