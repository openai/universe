from multiprocessing import pool
from universe import error, rewarder, vectorized

class Joint(vectorized.Wrapper):
    def __init__(self, env_m):
        self.env_m = env_m
        for env in self.env_m:
            if not env._configured:
                raise error.Error('Joint env should have been initialized: {}'.format(env))

        # TODO: generalize this. Doing so requires adding a vectorized
        # space mode.
        self.action_space = env_m[0].action_space
        self.observation_space = env_m[0].observation_space

        self.pool = pool.ThreadPool(min(len(env_m), 5))

    def _close(self):
        if hasattr(self, 'pool'):
            self.pool.close()

    @property
    def spec(self):
        return None

    @spec.setter
    def spec(self, value):
        pass

    def _render(self, mode='human', close=False):
        return self.env_m[0]._render(mode=mode, close=close)

    def _configure(self, **kwargs):
        self.n = sum(env.n for env in self.env_m)
        self.metadata = self.metadata.copy()
        self.metadata['render.modes'] = self.env_m[0].metadata['render.modes']

    def _reset(self):
        # Keep all env[0] action on the main thread, in case we ever
        # need to render. Otherwise we get segfaults from the
        # go-vncdriver.
        reset_m_async = self.pool.map_async(lambda env: env.reset(), self.env_m[1:])
        reset = self.env_m[0].reset()
        reset_m = [reset] + reset_m_async.get()

        observation_n = []
        for observation_m in reset_m:
            observation_n += observation_m
        return observation_n

    def _step(self, action_n):
        observation_n = []
        reward_n = []
        done_n = []
        info_n = []
        info = {}

        action_m = []
        for env in self.env_m:
            action_m.append(action_n[len(action_m):len(action_m)+env.n])

        # Keep all env[0] action on the main thread, in case we ever
        # need to render. Otherwise we get segfaults from the
        # go-vncdriver.
        step_m_async = self.pool.map_async(lambda arg: arg[0].step(arg[1]), zip(self.env_m[1:], action_m[1:]))
        step = self.env_m[0].step(action_m[0])
        step_m = [step] + step_m_async.get()

        for observation_m, reward_m, done_m, _info in step_m:
            observation_n += observation_m
            reward_n += reward_m
            done_n += done_m

            # copy in any info keys
            rewarder.merge_infos(info, _info)
            info_n += _info['n']

        info['n'] = info_n
        return observation_n, reward_n, done_n, info
