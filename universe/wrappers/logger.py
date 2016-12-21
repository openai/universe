# -*- coding: utf-8 -*-

import logging
import numpy as np
import six
import time

from universe import vectorized
from universe.utils import display

logger = logging.getLogger(__name__)
extra_logger = logging.getLogger('universe.extra.'+__name__)

def stats(count):
    flat = [e for vals in count for e in vals]
    if len(flat) == 0:
        return '(empty)'
    s = '%0.1f±%0.1f' % (np.mean(flat), np.std(flat))
    if six.PY2:
        # There is not a great way to backport Unicode support to Python 2.
        # We don't use it much, anyway. Easier just not to try.
        s = s.replace('±', '+-')
    return s

class Logger(vectorized.Wrapper):
    def __init__(self, env):
        super(Logger, self).__init__(env)

    def _configure(self, print_frequency=5, **kwargs):
        self.print_frequency = print_frequency
        extra_logger.info('Running VNC environments with Logger set to print_frequency=%s. To change this, pass "print_frequency=k" or "print_frequency=None" to "env.configure".', self.print_frequency)
        super(Logger, self)._configure(**kwargs)
        self._clear_step_state()
        self.metadata['render.modes'] = self.env.metadata['render.modes']

        self._last_step_time = None

    def _clear_step_state(self):
        self.frames = 0
        self.last_print = time.time()
        # time between action being sent and processed
        self.action_lag_n = [[] for _ in range(self.n)]
        # time between observation being generated on the server and being passed to add_metadata
        self.observation_lag_n = [[] for _ in range(self.n)]
        # time between observation being passed to add_metadata and being returned to Logger
        self.processing_lag = []
        # time between observation being returned by Logger and then action being passed to Throttle
        self.thinking_lag = []

        self.vnc_updates_n = [[] for _ in range(self.n)]
        self.vnc_bytes_n = [[] for _ in range(self.n)]
        self.vnc_pixels_n = [[] for _ in range(self.n)]
        self.reward_count_n = [[] for _ in range(self.n)]
        self.reward_total_n = [[] for _ in range(self.n)]
        self.reward_lag_n = [[] for _ in range(self.n)]
        self.rewarder_message_lag_n = [[] for _ in range(self.n)]

    def _step(self, action_n):
        observation_n, reward_n, done_n, info = self.env.step(action_n)
        if self.print_frequency is None:
            return observation_n, reward_n, done_n, info

        last_step_time = self._last_step_time
        self._last_step_time = time.time()

        # Printing
        self.frames += 1
        delta = time.time() - self.last_print
        if delta > self.print_frequency:
            fps = self.frames/delta

            # Displayed independently
            # action_lag = ','.join([diagnostics.display_timestamps_pair_max(action_lag) for action_lag in self.action_lag_n])
            # observation_lag = ','.join([diagnostics.display_timestamps_pair_max(observation_lag) for observation_lag in self.observation_lag_n])

            flat = False

            # Smooshed together
            action_lag, action_data = display.compute_timestamps_pair_max(self.action_lag_n, flat=flat)
            observation_lag, observation_data = display.compute_timestamps_pair_max(self.observation_lag_n, flat=flat)
            processing_lag, processing_data = display.compute_timestamps_sigma(self.processing_lag)
            thinking_lag, thinking_data = display.compute_timestamps_sigma(self.thinking_lag)
            reward_count = [sum(r) / delta for r in self.reward_count_n]
            if flat and len(reward_count) > 0:
                reward_count = np.mean(reward_count)
            reward_total = [sum(r) / delta for r in self.reward_total_n]
            if flat and len(reward_total) > 0:
                reward_total = np.mean(reward_total)
            reward_lag, reward_data = display.compute_timestamps_pair_max(self.reward_lag_n, flat=flat)
            rewarder_message_lag, rewarder_message_data = display.compute_timestamps_pair_max(self.rewarder_message_lag_n, flat=flat)
            vnc_updates_count = [sum(v) / delta for v in self.vnc_updates_n]
            if flat and len(vnc_updates_count) > 0:
                vnc_updates_count = np.mean(vnc_updates_count)

            # Always aggregate these ones
            if len(self.vnc_bytes_n) > 0:
                vnc_bytes_count = np.sum(e for vnc_bytes in self.vnc_bytes_n for e in vnc_bytes) / delta
            else:
                vnc_bytes_count = None
            if len(self.vnc_pixels_n) > 0:
                vnc_pixels_count = np.sum(e for vnc_pixels in self.vnc_pixels_n for e in vnc_pixels) / delta
            else:
                vnc_pixels_count = None

            reward_stats = stats(self.reward_count_n)
            vnc_updates_stats = stats(self.vnc_updates_n)
            vnc_bytes_stats = stats(self.vnc_bytes_n)
            vnc_pixels_stats = stats(self.vnc_pixels_n)

            reaction_time = []
            for a, o in zip(action_data, observation_data):
                try:
                    value = thinking_data['mean'] + processing_data['mean'] + a['mean'] + o['mean']
                except KeyError:
                    reaction_time.append(None)
                else:
                    reaction_time.append(display.display_timestamp(value))

            log = []
            for key, spec, value in [
                    ('vnc_updates_ps', '%0.1f', vnc_updates_count),
                    ('n', '%s', self.n),
                    ('reaction_time', '%s', reaction_time),
                    ('observation_lag', '%s', observation_lag),
                    ('action_lag', '%s', action_lag),
                    ('processing_lag', '%s', processing_lag),
                    ('thinking_lag', '%s', thinking_lag),
                    ('reward_ps', '%0.1f', reward_count),
                    ('reward_total', '%0.1f', reward_total),
                    ('vnc_bytes_ps[total]', '%0.1f', vnc_bytes_count),
                    ('vnc_pixels_ps[total]', '%0.1f', vnc_pixels_count),
                    ('reward_lag', '%s', reward_lag),
                    ('rewarder_message_lag', '%s', rewarder_message_lag),
                    ('fps', '%0.2f', fps),
            ]:
                if value == None:
                    continue

                if isinstance(value, list):
                    value = ','.join(spec % v for v in value)
                else:
                    value = spec % value
                log.append('%s=%s' % (key, value))

            if not log:
                log.append('(empty)')

            if self.frames != 0:
                logger.info('Stats for the past %.2fs: %s', delta, ' '.join(log))
            self._clear_step_state()

        # These are properties of the step rather than any one index
        observation_available_at = info.get('throttle.observation.available_at')
        if observation_available_at is not None:
            # (approximate time that we're going to return -- i.e. now, assuming Logger is fast)
            # - (time that the observation was passed to add_metadata)
            self.processing_lag.append(self._last_step_time - observation_available_at)

        action_available_at = info.get('throttle.action.available_at')
        if action_available_at is not None and last_step_time is not None:
            # (time that the action was generated) - (approximate time that we last returned)
            self.thinking_lag.append(action_available_at - last_step_time)

        # Saving of lags
        for i, info_i in enumerate(info['n']):
            observation_lag = info_i.get('stats.gauges.diagnostics.lag.observation')
            if observation_lag is not None:
                self.observation_lag_n[i].append(observation_lag)

            action_lag = info_i.get('stats.gauges.diagnostics.lag.action')
            if action_lag is not None:
                self.action_lag_n[i].append(action_lag)

            reward_count = info_i.get('reward.count')
            if reward_count is not None:
                self.reward_count_n[i].append(reward_count)

            reward_total = reward_n[i]
            if reward_total is not None:
                self.reward_total_n[i].append(reward_total)

            assert 'vnc.updates.n' not in info, 'Looks like you are using an old go-vncdriver. Please update to >=0.4.0: pip install --ignore-installed --no-cache-dir go-vncdriver'

            vnc_updates = info_i.get('stats.vnc.updates.n')
            if vnc_updates is not None:
                self.vnc_updates_n[i].append(vnc_updates)

            vnc_bytes = info_i.get('stats.vnc.updates.bytes')
            if vnc_bytes is not None:
                self.vnc_bytes_n[i].append(vnc_bytes)

            vnc_pixels = info_i.get('stats.vnc.updates.pixels')
            if vnc_pixels is not None:
                self.vnc_pixels_n[i].append(vnc_pixels)

            reward_lag = info_i.get('stats.gauges.diagnostics.lag.reward')
            if reward_lag is not None:
                self.reward_lag_n[i].append(reward_lag)

            rewarder_message_lag = info_i.get('stats.gauges.diagnostics.lag.rewarder_message')
            if rewarder_message_lag is not None:
                self.rewarder_message_lag_n[i].append(rewarder_message_lag)

        return observation_n, reward_n, done_n, info
