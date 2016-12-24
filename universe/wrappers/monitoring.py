import logging

from gym import monitoring
from universe.vectorized import core  # Cannot import vectorized directly without inducing a cycle

logger = logging.getLogger(__name__)

def Monitor(directory, video_callable=None, force=False, resume=False,
            write_upon_reset=False, uid=None, mode=None):
    class Monitor(core.Wrapper):
        def _configure(self, **kwargs):
            super(Monitor, self)._configure(**kwargs)

            # We have to wait until configure to set the monitor because we need the number of instances in a vectorized env
            self._start_monitor()

        def _start_monitor(self):
            # Circular dependencies :(
            from universe import wrappers
            # We need to maintain pointers to these to avoid them being
            # GC'd. They have a weak reference to us to avoid cycles.
            self._unvectorized_envs = [wrappers.WeakUnvectorize(self, i) for i in range(self.n)]

            # For now we only monitor the first env
            self._monitor = monitoring.MonitorManager(self._unvectorized_envs[0])
            self._monitor.start(directory, video_callable, force, resume,
                                write_upon_reset, uid, mode)

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
            super(Monitor, self)._close()
            self._monitor.close()

        def set_monitor_mode(self, mode):
            logger.info("Setting the monitor mode is deprecated and will be removed soon")
            self._monitor._set_mode(mode)

    return Monitor
