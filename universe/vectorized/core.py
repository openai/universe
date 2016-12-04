import gym
from gym import spaces
from universe import error

class Env(gym.Env):
    metadata = {
        'runtime.vectorized': True
    }

    # User should set this!
    n = None

    @property
    def monitor(self):
        if not self.metadata['runtime.vectorized']:
            # Just delegate if we're not actually vectorized (like
            # Unvectorize)
            return super(Env, self).monitor

        if not hasattr(self, '_monitor'):
            # Not much we can do if we don't know how wide we'll
            # be. This can happen when closing.
            if self.n is None:
                raise error.Error('You must call "configure()" before accesssing the monitor for {}'.format(self))

            # Circular dependencies :(
            from universe import wrappers
            from universe.vectorized import monitoring
            # We need to maintain pointers to these to them being
            # GC'd. They have a weak reference to us to avoid cycles.
            self._unvectorized = [wrappers.WeakUnvectorize(self, i) for i in range(self.n)]
            # Store reference to avoid GC
            # self._render_cached = monitoring.RenderCache(self)
            self._monitor = monitoring.Monitor(self._unvectorized)
        return self._monitor

class Wrapper(Env, gym.Wrapper):
    autovectorize = True
    standalone = True

    def __init__(self, env=None):
        super(Wrapper, self).__init__(env)
        if env is not None and not env.metadata.get('runtime.vectorized'):
            if self.autovectorize:
                # Circular dependency :(
                from universe import wrappers
                env = wrappers.Vectorize(env)
            else:
                raise error.Error('This wrapper can only wrap vectorized envs (i.e. where env.metadata["runtime.vectorized"] = True), not {}. Set "self.autovectorize = True" to automatically add a Vectorize wrapper.'.format(env))

        if env is None and not self.standalone:
            raise error.Error('This env requires a non-None env to be passed. Set "self.standalone = True" to allow env to be omitted or None.')

        self.env = env

    def _configure(self, **kwargs):
        super(Wrapper, self)._configure(**kwargs)
        assert self.env.n is not None, "Did not set self.env.n: self.n={} self.env={} self={}".format(self.env.n, self.env, self)
        self.n = self.env.n

class ObservationWrapper(Wrapper, gym.ObservationWrapper):
    pass

class RewardWrapper(Wrapper, gym.RewardWrapper):
    pass

class ActionWrapper(Wrapper, gym.ActionWrapper):
    pass
