import logging
import numpy as np
from universe import vectorized
from universe.wrappers import render

logger = logging.getLogger(__name__)

def WrappedMultiprocessingEnv(env_id):
    return render.Render(EpisodeID(vectorized.MultiprocessingEnv(env_id)))

class RemoveNones(vectorized.Wrapper):
    """The vectorized environment will return None for any indexes that
    have already exceeded their episode count (not to be confused with
    the Nones returned by resetting environments in the real-time
    case). For convenience, we instead return a plausible observation
    in each such slot.
    """
    def __init__(self, env):
        super(RemoveNones, self).__init__(env)
        self.plausible_observation = None

    def _reset(self):
        observation_n = self.env.reset()
        self.plausible_observation = observation_n[0]
        return observation_n

    def _step(self, action_n):
        observation_n, reward_n, done_n, info = self.env.step(action_n)
        observation_n = [ob if ob is not None else self.plausible_observation for ob in observation_n]
        return observation_n, reward_n, done_n, info

class EpisodeID(vectorized.Wrapper):
    """
For each episode, return its id, and also return the total number of contiguous 
episodes that are now done. 
"""
    def _configure(self, episode_limit=None, **kwargs):
        super(EpisodeID, self)._configure(**kwargs)
        assert self.metadata.get('runtime.vectorized')
        self.episode_limit = episode_limit
        self._clear_state()

    def _clear_state(self):
        self.done_to = -1
        self.extra_done = set()
        self.episode_ids = list(range(self.n))

    def _set_done_to(self):
        while True:
            next_done_to = self.done_to + 1
            if next_done_to in self.extra_done:
                self.done_to = next_done_to
                self.extra_done.remove(next_done_to)
            else:
                break

    def _reset(self):
        self._clear_state()
        return self.env.reset()

    def _step(self, action_n):
        observation_n, reward_n, done_n, info = self.env.step(action_n)
        # Pass along ID of potentially-done episode
        for i, info_i in enumerate(info['n']):
            info_i['vectorized.episode_id'] = self.episode_ids[i]

        done_i = np.argwhere(done_n).reshape(-1)
        if len(done_i):
            for i in done_i:
                self.extra_done.add(self.episode_ids[i])
                # Episode completed, so we bump its value
                self.episode_ids[i] += self.n
                if self.episode_limit is not None and self.episode_ids[i] > self.episode_limit:
                    logger.debug('Masking: index=%s episode_id=%s', i, self.episode_ids[i])
                    self.env.mask(i)
            self._set_done_to()

        # Pass along the number of contiguous episodes that are now done
        info['vectorized.done_to'] = self.done_to
        return observation_n, reward_n, done_n, info
