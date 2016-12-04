import logging
from universe import vectorized

logger = logging.getLogger(__name__)

class Vision(vectorized.Wrapper):
    """
At present, an observation from a vectorized universe environment returns a list of 
dicts. Each dict contains input data for each modality.  Modalities include 'vision'
and 'text', and it is possible to add other modalities in the future (such as 'audio').

The Vision wrapper extracts the vision modality and discards all others.  This is convenient
when we only care about the visual input.
"""

    def _reset(self):
        observation_n = self.env.reset()
        return [ob['vision'] if ob is not None else ob for ob in observation_n]

    def _step(self, action_n):
        observation_n, reward_n, done_n, info_n = self.env.step(action_n)
        observation_n = [ob['vision'] if ob is not None else ob for ob in observation_n]
        return observation_n, reward_n, done_n, info_n
