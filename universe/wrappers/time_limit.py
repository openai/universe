import logging

import time
from universe import pyprofile, vectorized

logger = logging.getLogger(__name__)

DEFAULT_MAX_EPISODE_SECONDS = 20 * 60.  # Default to 20 minutes if there is no explicit limit


class TimeLimit(vectorized.Wrapper):
    def _configure(self, **kwargs):
        super(TimeLimit, self)._configure(**kwargs)
        self.max_episode_seconds = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_seconds', None)
        self.max_episode_steps = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps', None)

        if self.max_episode_seconds is None and self.max_episode_steps is None:
            self.max_episode_seconds = DEFAULT_MAX_EPISODE_SECONDS

        self._elapsed_steps = 0
        self._episode_started_at = None

    @property
    def _elapsed_seconds(self):
        return time.time() - self._episode_started_at

    def _past_limit(self):
        """Return true if we are past our limit"""
        if self.max_episode_steps is not None and self.max_episode_steps < self._elapsed_steps:
            return True

        if self.max_episode_seconds is not None and self.max_episode_seconds < self._elapsed_seconds:
            return True

        return False

    def _step(self, action_n):
        observation_n, reward_n, done_n, info = self.env.step(action_n)

        self._elapsed_steps += 1
        if self._episode_started_at is None:
            self._episode_started_at = time.time()

        if self._past_limit():
            observation_n = self.env.reset()  # Force a reset
            return observation_n, reward_n, [True] * self.n, info  # Return the new observation and done = True
        else:
            return observation_n, reward_n, done_n, info

    def _reset(self):
        self._episode_started_at = time.time()
        self._elapsed_steps = 0
        return self.env.reset()
