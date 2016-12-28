import logging

from gym import monitoring
from universe.vectorized import core  # Cannot import vectorized directly without inducing a cycle
from universe.wrappers.time_limit import TimeLimit

logger = logging.getLogger(__name__)

class _Monitor(core.Wrapper):
    def __init__(self, env, directory, video_callable=None, force=False,
                 resume=False, write_upon_reset=False, uid=None, mode=None):
        super(_Monitor, self).__init__(env)
        self.directory = directory
        self.video_callable = video_callable
        self.force = force
        self.resume = resume
        self.write_upon_reset = write_upon_reset
        self.uid = uid
        self.mode = mode

        # TODO if we want to monitor more than one instance in a vectorized
        # env we'll have to do this after configure()
        self._start_monitor()

    def _start_monitor(self):
        # Circular dependencies :(
        from universe import wrappers
        # We need to maintain pointers to these to avoid them being
        # GC'd. They have a weak reference to us to avoid cycles.
        # TODO if we want to monitor more than one instance in a vectorized
        # env we'll need to actually fix WeakUnvectorize
        self._unvectorized_envs = [wrappers.WeakUnvectorize(self, i) for i in range(1)]

        # For now we only monitor the first env
        self._monitor = monitoring.MonitorManager(self._unvectorized_envs[0])
        self._monitor.start(
            self.directory,
            self.video_callable,
            self.force,
            self.resume,
            self.write_upon_reset,
            self.uid,
            self.mode,
        )

    def _step(self, action_n):
        self._monitor._before_step(action_n[0])
        observation_n, reward_n, done_n, info = self.env.step(action_n)
        done_n[0] = self._monitor._after_step(observation_n[0], reward_n[0], done_n[0], info)
        return observation_n, reward_n, done_n, info

    def _reset(self):
        self._monitor._before_reset()
        observation_n = self.env.reset()
        self._monitor._after_reset(observation_n[0])
        return observation_n

    def _close(self):
        super(_Monitor, self)._close()
        self._monitor.close()

    def set_monitor_mode(self, mode):
        logger.info("Setting the monitor mode is deprecated and will be removed soon")
        self._monitor._set_mode(mode)

def Monitor(env, directory, video_callable=None, force=False, resume=False,
            write_upon_reset=False, uid=None, mode=None):
    return _Monitor(TimeLimit(env), directory, video_callable, force, resume,
                    write_upon_reset, uid, mode)
