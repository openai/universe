import logging
import time
from universe import pyprofile, rewarder, spaces, vectorized

logger = logging.getLogger(__name__)

class Throttle(vectorized.Wrapper):
    """
    A env wrapper that makes sending the action ASAP.

    Previous implementation would sleep first and then call env._step.
    This implementation calls env._step twice:
        1. first call submits given action
        2. after sleeping based on fps, second call submits empty action to
           receive observation.

    visual observation from first call is discarded.
    metadata and rewards from the two calls are merged.
    text observations are merged as well.
    """
    def __init__(self, env):
        super(Throttle, self).__init__(env)

        self._steps = None

    def _configure(self, skip_metadata=False, fps='default', **kwargs):
        super(Throttle, self)._configure(**kwargs)
        if fps == 'default':
            fps = self.metadata['video.frames_per_second']
        self.fps = fps
        self.skip_metadata = skip_metadata

        self.diagnostics = self.unwrapped.diagnostics

    def _reset(self):
        # We avoid aggregating reward/info across episode boundaries
        # by caching it on the object
        self._deferred_reward_n = None
        self._deferred_done_n = None
        self._deferred_info_n = None

        observation = self.env.reset()
        self._start_timer()
        return observation

    def _step(self, action_n):
        if self._steps is None:
            self._start_timer()
        self._steps += 1

        accum_observation_n, accum_reward_n, accum_done_n, accum_info = self._substep(action_n)
        accum_info['throttle.action.available_at'] = time.time()

        # Record which indexes we were just peeking at, so when we
        # make the follow-up we'll be sure to peek there too.
        peek_n = [any(spaces.PeekReward for peek in action) for action in action_n]

        if self.fps is None:
            return accum_observation_n, accum_reward_n, accum_done_n, accum_info

        accum_info['stats.throttle.sleep'] = 0
        while True:
            # See how much time we have to idle
            delta = self._start + 1./self.fps * self._steps - time.time()

            # The following assumes that our control loop
            if delta < 0:
                # We're out of time. Just get out of here.
                delta = abs(delta)
                if delta >= 1:
                    logger.info('Throttle fell behind by %.2fs; lost %.2f frames', delta, self.fps*delta)
                pyprofile.timing('vnc_env.Throttle.lost_sleep', delta)
                self._start_timer()
                break
            # elif delta < 0.008:
            #     # Only have 8ms. Let's spend it sleeping, and
            #     # return an image which may have up to an
            #     # additional 8ms lag.
            #     #
            #     # 8ms is reasonably arbitrary; we just want something
            #     # that's small where it's not actually going to help
            #     # if we make another step call. Step with 32 parallel
            #     # envs takes about 6ms (about half of which is
            #     # diagnostics, which could be totally async!), so 8 is
            #     # a reasonable choice for now..
            #     pyprofile.timing('vnc_env.Throttle.sleep', delta)
            #     accum_info['stats.throttle.sleep'] += delta
            #     time.sleep(delta)
            #     break
            else:
                # We've got plenty of time. Sleep for up to 16ms, and
                # then refresh our current frame. We need to
                # constantly be calling step so that our lags are
                # reported correctly, within 16ms. (The layering is
                # such that the vncdriver doesn't know which pixels
                # correspond to metadata, and the diagnostics don't
                # know when pixels first got painted. So we do our
                # best to present frames as they're ready to the
                # diagnostics.)
                delta = min(delta, 0.016)
                pyprofile.timing('vnc_env.Throttle.sleep', delta)
                accum_info['stats.throttle.sleep'] += delta
                time.sleep(delta)

                # We want to merge in the latest reward/done/info so that our
                # agent has the most up-to-date info post-sleep, but also want
                # to avoid popping any rewards where done=True (since we'd
                # have to merge across episode boundaries).
                action_n = []
                for done, peek in zip(accum_done_n, peek_n):
                    if done or peek:
                        # No popping of reward/done
                        action_n.append([spaces.PeekReward])
                    else:
                        action_n.append([])

                observation_n, reward_n, done_n, info = self._substep(action_n)

                # Merge observation, rewards and metadata.
                # Text observation has order in which the messages are sent.
                rewarder.merge_n(
                    accum_observation_n, accum_reward_n, accum_done_n, accum_info,
                    observation_n, reward_n, done_n, info,
                )

        return accum_observation_n, accum_reward_n, accum_done_n, accum_info

    def _substep(self, action_n):
        with pyprofile.push('vnc_env.Throttle.step'):
            start = time.time()
            # Submit the action ASAP, before the thread goes to sleep.
            observation_n, reward_n, done_n, info = self.env.step(action_n)

            available_at = info['throttle.observation.available_at'] = time.time()
            if available_at - start > 1:
                logger.info('env.step took a long time: %.2fs', available_at - start)
            if not self.skip_metadata and self.diagnostics is not None:
                # Run (slow) diagnostics
                self.diagnostics.add_metadata(observation_n, info['n'], available_at=available_at)
            return observation_n, reward_n, done_n, info

    def _start_timer(self):
        self._start = time.time()
        self._steps = 0
