import logging
import time
from universe import pyprofile, vectorized

logger = logging.getLogger(__name__)

class Timer(vectorized.Wrapper):
    """
Calcultae how much time was spent actually doing work.  Display result
via pyprofile.
"""

    def _reset(self):
        with pyprofile.push('vnc_env.Timer.reset'):
            return self.env.reset()

    def _step(self, action_n):
        start = time.time()
        with pyprofile.push('vnc_env.Timer.step'):
            observation_n, reward_n, done_n, info = self.env.step(action_n)

        # Calculate how much time was spent actually doing work
        sleep = info.get('stats.throttle.sleep')
        if sleep is None or sleep < 0:
            sleep = 0
        pyprofile.timing('vnc_env.Timer.step.excluding_sleep', time.time() - start - sleep)
        return observation_n, reward_n, done_n, info
