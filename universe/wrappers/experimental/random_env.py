import logging
from universe import vectorized

logger = logging.getLogger(__name__)

class RandomEnv(vectorized.Wrapper):
    '''
    Randomly sample from a list of env_ids between episodes.

    Passes a list of env_ids to configure. When done=True, calls env.reset()
    to sample from the list.
    '''
    def __init__(self, env, env_ids):
        super(RandomEnv, self).__init__(env)
        self.env_ids = env_ids

    def _configure(self, **kwargs):
        super(RandomEnv, self)._configure(sample_env_ids=self.env_ids, **kwargs)

    def _reset(self):
        observation_n = self.env.reset()
        return [ob['vision'] if ob is not None else ob for ob in observation_n]

    def _step(self, action_n):
        assert self.n == 1
        observation, reward, done, info = self.env.step(action_n)
        if any(done):
            self.env.reset()
        return observation, reward, done, info
