import logging
import threading

logger = logging.getLogger()

def parse_episode_id(episode_id):
    if episode_id is None:
        return -1
    return int(episode_id)

def generate_episode_id(parsed):
    if parsed == -1:
        return None
    return str(parsed)

def compare_ids(a, b):
    if a == b:
        return 0
    elif a is None:
        return -1
    elif b is None:
        return 1
    elif parse_episode_id(a) < parse_episode_id(b):
        return -1
    else:
        return 1

class EnvStatus(object):
    def __init__(self, label=None, primary=True):
        self.cv = threading.Condition()
        self._env_id = None
        self._env_state = None
        self._episode_id = '0'
        self._fps = None
        self.label = label or 'EnvStatus'
        self.primary = primary

    def env_info(self):
        with self.cv:
            return {
                'env_state': self._env_state,
                'env_id': self._env_id,
                'episode_id': self._episode_id,
                'fps': self._fps,
            }

    def set_env_info(self, env_state=None, env_id=None, episode_id=None, bump_past=None, fps=None):
        """Atomically set the environment state tracking variables.
        """
        with self.cv:
            if env_id is None:
                env_id = self._env_id
            if env_state is None:
                env_state = self._env_state
            if fps is None:
                fps = self._fps
            self.cv.notifyAll()

            old_episode_id = self._episode_id
            if self.primary:
                current_id = parse_episode_id(self._episode_id)
                # Bump when changing from resetting -> running
                if bump_past is not None:
                    bump_past_id = parse_episode_id(bump_past)
                    current_id = max(bump_past_id+1, current_id+1)
                elif env_state == 'resetting':
                    current_id += 1
                self._episode_id = generate_episode_id(current_id)
                assert self._fps or fps
            elif episode_id is False:
                # keep the same episode_id: this is just us proactive
                # setting the state to resetting after a done=True
                pass
            else:
                assert episode_id is not None, "No episode_id provided. This likely indicates a misbehaving server, which did not send an episode_id"
                self._episode_id = episode_id
            self._fps = fps
            logger.info('[%s] Changing env_state: %s (env_id=%s) -> %s (env_id=%s) (episode_id: %s->%s, fps=%s)', self.label, self._env_state, self._env_id, env_state, env_id, old_episode_id, self._episode_id, self._fps)
            self._env_state = env_state
            if env_id is not None:
                self._env_id = env_id

            return self.env_info()

    @property
    def episode_id(self):
        with self.cv:
            return self._episode_id

    @property
    def env_state(self):
        with self.cv:
            return self._env_state

    @env_state.setter
    def env_state(self, value):
        # TODO: Validate env_state
        self.set_env_info(value)

    @property
    def env_id(self):
        with self.cv:
            return self._env_id

    @env_id.setter
    def env_id(self, value):
        self.set_env_info(None, env_id=value)

    @property
    def fps(self):
        with self.cv:
            return self._fps

    def wait_for_env_state_change(self, start_state):
        with self.cv:
            while True:
                if self._env_state != start_state:
                    return self.env_info()
                self.cv.wait(timeout=10)
