import logging
import threading
import time

from universe import error
from universe.rewarder import env_status, merge

logger = logging.getLogger(__name__)
extra_logger = logging.getLogger('universe.extra.'+__name__)

class RewardState(object):
    def __init__(self, label, episode_id):
        self.label = label

        self.count = 0
        self.reward = 0.
        self.text = []
        self.done = False
        self.info = {}
        self._observation = None

        self._episode_id = episode_id
        self._env_state = None

    def push_time(self, remote_time, local_time):
        # Sometimes helpful diagnostic info
        self.info['reward_buffer.remote_time'] = remote_time
        self.info['reward_buffer.local_time'] = local_time

    def set_env_info(self, env_state):
        self._env_state = env_state

    def push_text(self, text):
        self.text.append(text)

    def push_done(self, done, info):
        # Consider yourself done whenever a reward crosses episode
        # boundaries.
        self.done = self.done or done
        if done:
            info['reward_buffer.done_time'] = time.time()
        self.push_info(info)

    def push_info(self, info):
        merge.merge_infos(self.info, info)

    def push(self, reward, done, info):
        # extra_logger.debug('[%s] RewardState: pushing reward %s to episode_id %s', self.label, reward, self._episode_id)
        self.count += 1
        self.reward += reward

        # Consider yourself done whenever a reward crosses episode
        # boundaries.
        self.push_done(done, info)

    def pop_text(self):
        text = self.text
        self.text = []
        return text

    def pop_info(self):
        info = self.info
        self.info = {}
        info['env.text'] = self.pop_text()
        if self._observation is not None:
            # Only used for the debugging gym-core envs with
            # rewarder_observation set.
            info['rewarder.observation'] = (self._observation, self._episode_id)
        info['env_status.episode_id'] = self._episode_id
        info['env_status.env_state'] = self._env_state
        return info

    def pop(self):
        info = self.pop_info()

        count = self.count
        reward = self.reward
        done = self.done

        self.count = 0
        self.reward = 0.
        self.done = False

        info['stats.reward.count'] = count
        extra_logger.debug('[%s] RewardState: popping reward %s from episode_id %s', self.label, reward, self._episode_id)
        return reward, done, info

    def set_observation(self, observation):
        self._observation = observation

# Buffers up incoming rewards
class RewardBuffer(object):
    def __init__(self, label):
        self.cv = threading.Condition()
        self.label = label

        self._current_episode_id = None
        self._reward_state = {}

        self._masked = True

        self._remote_env_state = None
        self._remote_env_id = None
        self._remote_episode_id = None
        self._remote_fps = None

    def reward_state(self, episode_id):
        try:
            return self._reward_state[episode_id]
        except:
            extra_logger.info('[%s] RewardBuffer: Creating new RewardState for episode_id=%s', self.label, episode_id)
            reward_state = self._reward_state[episode_id] = RewardState(self.label, episode_id)
            if self._current_episode_id is None and not self._masked:
                extra_logger.info('[%s] RewardBuffer advancing: No active episode, so activating episode_id=%s', self.label, episode_id)
                self._current_episode_id = episode_id
                self._drop_below(episode_id)
            if episode_id is not None and not self._masked:
                # If we're masked we'll be dropping everything below the reset ID anyway
                valid = self._valid_ids()
                for id in valid:
                    if id == episode_id:
                        continue
                    state = self._reward_state[id]
                    if state.done:
                        continue
                    extra_logger.info('[%s] RewardBuffer received message for episode_id=%s but no done=True message for %s. Artificially marking %s as done=True.', self.label, episode_id, id, id)
                    state.push_done(True, {'env_status.artificial.done': True})

            return reward_state

    def set_env_info(self, env_state, env_id, episode_id, fps):
        with self.cv:
            if self._remote_env_state is not None:
                extra_logger.info('[%s] RewardBuffer changing env_state: %s (env_id=%s) -> %s (env_id=%s) (episode_id: %s->%s, fps=%s, masked=%s, current_episode_id=%s)', self.label, self._remote_env_state, self._remote_env_id, env_state, env_id, self._remote_episode_id, episode_id, fps, self._masked, self._current_episode_id)
            else:
                extra_logger.info('[%s] RewardBuffer: Initial env_state: %s (env_id=%s) (episode_id: %s, fps=%s, masked=%s, current_episode_id=%s)', self.label, env_state, env_id, episode_id, fps, self._masked, self._current_episode_id)

            self._remote_env_state = env_state
            self._remote_env_id = env_id
            self._remote_episode_id = episode_id
            self._remote_fps = fps

            self.reward_state(episode_id).set_env_info(env_state)

    def set_observation(self, episode_id, observation):
        with self.cv:
            self.reward_state(episode_id).set_observation(observation)

    def push_time(self, episode_id, remote_time, local_time):
        with self.cv:
            self.reward_state(episode_id).push_time(remote_time, local_time)

    def push_text(self, episode_id, text):
        with self.cv:
            self.reward_state(episode_id).push_text(text)
            self.cv.notifyAll()

    def push_info(self, episode_id, info):
        # Just send some info
        with self.cv:
            self.reward_state(episode_id).push_info(info)

    def push(self, episode_id, reward, done, info):
        with self.cv:
            self.reward_state(episode_id).push(reward, done, info)
            self.cv.notifyAll()

    def pop(self, peek=False):
        with self.cv:
            self.cv.notifyAll()
            if peek:
                # This happens when a higher layer wants to poll for
                # new observations being ready, but doesn't want to
                # pop any rewards.
                max_id = self._max_id()
                reward_state = self.reward_state(self._current_episode_id)
                peek_state = self.reward_state(max_id)
                peek_id = peek_state._episode_id
                peek_state = peek_state._env_state
                if self._masked:
                    assert reward_state._episode_id is None
                    assert reward_state._env_state is None
                    peek_id = None
                    peek_state = None
                return 0, False, {
                    'peek': True,

                    'env_status.episode_id': reward_state._episode_id,
                    'env_status.env_state': reward_state._env_state,

                    'env_status.peek.episode_id': peek_id,
                    'env_status.peek.env_state': peek_state,
                }

            reward, done, info = self.reward_state(self._current_episode_id).pop()
            if done:
                # We return the *observation* from the new,
                # reward/done from the old, and a merged info with
                # keys from the new taking precedence.
                self._advance()

                new_state = self.reward_state(self._current_episode_id)
                try:
                    info['env_status.complete.episode_id'] = info['env_status.episode_id']
                except KeyError:
                    pass
                try:
                    info['env_status.complete.env_state'] = info['env_status.env_state']
                except KeyError:
                    pass
                info['env_status.episode_id'] = new_state._episode_id
                info['env_status.env_state'] = new_state._env_state
                new_text = self.reward_state(self._current_episode_id).pop_text()
                if len(info['env.text']) > 0:
                    extra_logger.info('[%s] RewardBuffer dropping env.text for completed episode %s: %s', self.label, info['env_status.episode_id'], info['env.text'])
                info['env.text'] = new_text
            return reward, done, info

    def mask(self):
        with self.cv:
            extra_logger.info('[%s] RewardBuffer advancing: masking until reset completes; setting current_episode_id=None', self.label)
            self._masked = True
            self._current_episode_id = None

    def reset(self, episode_id):
        with self.cv:
            extra_logger.info('[%s] RewardBuffer advancing: unmasking after explicit reset: episode_id=%s', self.label, episode_id)
            self._masked = False
            self._drop_below(episode_id, quiet=True)
            self._current_episode_id = episode_id
            self.push_info(episode_id, {'env_status.reset.episode_id': episode_id})

    def _max_id(self):
        valid_ids = self._valid_ids()
        if len(valid_ids) > 0:
            parsed = max(env_status.parse_episode_id(k) for k in valid_ids)
            return env_status.generate_episode_id(parsed)
        else:
            return None

    def _valid_ids(self):
        return [r for r in self._reward_state.keys() if r is not None]

    def _advance(self):
        completed_episode_id = self._current_episode_id
        del self._reward_state[completed_episode_id]

        if None in self._reward_state:
            extra_logger.warn('[%s] WARNING: RewardBuffer: while advancing from %s, None was in reward state: %s', self.label, completed_episode_id, self._reward_state)

        max_id = self._max_id()
        if max_id is not None:
            self._current_episode_id = max_id
            if env_status.compare_ids(completed_episode_id, self._current_episode_id) >= 0:
                extra_logger.info("[%s] RewardBuffer advancing: setting episode_id=None until new data received. Rare condition reached where message for old environment received after new one: completed_episode_id=%r self._current_episode_id=%r (%r). This is ok, but something we may want to fix in the future", self.label, completed_episode_id, self._current_episode_id, self._reward_state)
                self._current_episode_id = None
            else:
                extra_logger.info('[%s] RewardBuffer advancing: has data for next episode: %s->%s', self.label, completed_episode_id, self._current_episode_id)
                self._drop_below(self._current_episode_id)
        else:
            extra_logger.info('[%s] RewardBuffer advancing: setting episode_id=None until new data received (was episode_id=%s)', self.label, completed_episode_id)
            self._current_episode_id = None

    def _drop_below(self, episode_id, quiet=False):
        dropped = set()
        for stored_id in self._reward_state:
            if env_status.compare_ids(stored_id, episode_id) < 0:
                dropped.add(stored_id)

        if len(dropped) > 0:
            if quiet:
                log = extra_logger.debug
            else:
                log = extra_logger.info
            log('[%s] RewardBuffer: dropping stale episode data: dropped=%s episode_id=%s', self.label, dropped, episode_id)
        for stored_id in dropped:
            del self._reward_state[stored_id]

    def wait_for_step(self, error_buffer=None, timeout=None):
        # TODO: this might be cleaner using channels
        with self.cv:
            start = time.time()
            while True:
                if self.count != 0:
                    return
                elif timeout is not None and time.time() - start > timeout:
                    raise error.Error('No rewards received in {}s'.format(timeout))

                if error_buffer:
                    error_buffer.check()

                self.cv.wait(timeout=0.5)
