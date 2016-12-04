from universe import rewarder, spaces, vectorized

class BlockingReset(vectorized.Wrapper):
    """
By default, a reset in universe is not a blocking operation.  This 
wrapper changes it. 
"""

    def __init__(self, *args, **kwargs):
        super(BlockingReset, self).__init__(*args, **kwargs)
        self.reward_n = None
        self.done_n = None
        self.info = None

    def _reset(self):
        observation_n = self.env.reset()
        self.reward_n = [0] * self.n
        self.done_n = [False] * self.n
        self.info = {'n': [{} for _ in range(self.n)]}

        while any(ob is None for ob in observation_n):
            action_n = []
            for done in self.done_n:
                if done:
                    # No popping of reward/done. Don't want to merge across episode boundaries.
                    action_n.append([spaces.PeekReward])
                else:
                    action_n.append([])
            new_observation_n, new_reward_n, new_done_n, new_info = self.env.step(action_n)
            rewarder.merge_n(
                observation_n, self.reward_n, self.done_n, self.info,
                new_observation_n, new_reward_n, new_done_n, new_info
            )
        return observation_n

    def _step(self, action_n):
        observation_n, reward_n, done_n, info = self.env.step(action_n)
        if self.reward_n is not None:
            rewarder.merge_n(
                observation_n, reward_n, done_n, info,
                [None] * self.n, self.reward_n, self.done_n, self.info
            )
            self.reward_n = self.done_n = self.info = None

        while any(ob is None for ob in observation_n):
            action_n = []
            for done in done_n:
                if done:
                    # No popping of reward/done. Don't want to merge across episode boundaries.
                    action_n.append([spaces.PeekReward])
                else:
                    action_n.append([])
            new_observation_n, new_reward_n, new_done_n, new_info = self.env.step(action_n)
            rewarder.merge_n(
                observation_n, reward_n, done_n, info,
                new_observation_n, new_reward_n, new_done_n, new_info
            )
        return observation_n, reward_n, done_n, info
