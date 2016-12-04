import gym
import logging
from universe import rewarder, spaces, vectorized

logger = logging.getLogger(__name__)

class GymCoreSync(vectorized.Wrapper):
    """A synchronized version of the core envs. Its semantics should match
    that of the core envs. (By default, observations are pixels from
    the VNC session, but it also supports receiving the normal Gym
    observations over the rewarder protocol.)

    Provided primarily for testing and debugging.
    """

    def __init__(self, env):
        super(GymCoreSync, self).__init__(env)
        self.reward_n = None
        self.done_n = None
        self.info = None

        # Metadata has already been cloned
        self.metadata['semantics.async'] = False

    def _reset(self):
        observation_n = self.env.reset()
        new_observation_n, self.reward_n, self.done_n, self.info = self.env.step([[] for i in range(self.n)])
        rewarder.merge_observation_n(observation_n, new_observation_n)

        # Fast forward until the observation is caught up with the rewarder
        self._flip_past(observation_n, self.reward_n, self.done_n, self.info)

        assert all(r == 0 for r in self.reward_n), "Unexpectedly received rewards during reset phase: {}".format(self.reward_n)
        return observation_n

    def _step(self, action_n):
        # Add C keypress in order to "commit" the action, as
        # interpreted by the remote.
        action_n = [action + [
            spaces.KeyEvent.by_name('c', down=True),
            spaces.KeyEvent.by_name('c', down=False)
        ] for action in action_n]

        observation_n, reward_n, done_n, info = self.env.step(action_n)
        if self.reward_n is not None:
            rewarder.merge_n(
                observation_n, reward_n, done_n, info,
                [None] * self.n, self.reward_n, self.done_n, self.info,
            )
            self.reward_n = self.done_n = self.info = None

        while True:
            count = len([True for info_i in info['n'] if info_i['stats.reward.count'] == 0])
            if count > 0:
                logger.debug('[GymCoreSync] Still waiting on %d envs to receive their post-commit reward', count)
            else:
                break

            new_observation_n, new_reward_n, new_done_n, new_info = self.env.step([[] for i in range(self.n)])
            rewarder.merge_n(
                observation_n, reward_n, done_n, info,
                new_observation_n, new_reward_n, new_done_n, new_info
            )

        assert all(info_i['stats.reward.count'] == 1 for info_i in info['n']), "Expected all stats.reward.counts to be 1: {}".format(info)

        # Fast forward until the observation is caught up with the rewarder
        self._flip_past(observation_n, reward_n, done_n, info)
        return observation_n, reward_n, done_n, info

    def _flip_past(self, observation_n, reward_n, done_n, info):
        # Wait until all observations are past the corresponding reset times
        remote_target_time = [info_i['reward_buffer.remote_time'] for info_i in info['n']]
        while True:
            new_observation_n, new_reward_n, new_done_n, new_info = self.env.step([[] for i in range(self.n)])

            # info_i.get['diagnostics.image_remote_time'] may not exist, for example when an env
            # is resetting. target is a timestamp, thus > 0, so these will count as "need to catch up"
            deltas = [target - info_i.get('diagnostics.image_remote_time', 0) for target, info_i in zip(remote_target_time, new_info['n'])]
            count = len([d for d in deltas if d > 0])

            rewarder.merge_n(
                observation_n, reward_n, done_n, info,
                new_observation_n, new_reward_n, new_done_n, new_info
            )

            if count == 0:
                return
            else:
                logger.debug('[GymCoreSync] Still waiting on %d envs to catch up to their targets: %s', count, deltas)
