import logging
import os, time
import pytest
import numpy as np
import gym, universe
from universe import wrappers
from universe.vncdriver import constants
from universe.spaces import vnc_event

logger = logging.getLogger(__name__)


def create_dd_env(remotes=None, client_id=None):
    env = gym.make('flashgames.DuskDrive-v0')
    assert env.metadata['runtime.vectorized']

    env = wrappers.Logger(env)
    env = wrappers.BlockingReset(env)
    env = wrappers.Vision(env)
    env = wrappers.EpisodeID(env)
    env = wrappers.Unvectorize(env)

    env.configure(remotes=remotes, fps=10, observer=True, client_id=client_id)

    return env

@pytest.mark.skip()
def test_multi_env():
    """
    Create 2 envs pointing at the same VNC server and alternate using them.
    The vnc-agent eval workers do this.
    It's nontrivial because the envs have to start rejecting updates when they're not active, and start
    accepting again.
    You should see logs like
        update queue max of 60 reached; pausing further updates
    after every switch.
    You can watch it on a second VNC client. You should see the mouse slowly circling the screen.
    """
    loops = 1

    logger.info('test_multi_env, loops=%d', loops)
    e2 = create_dd_env(1, 'test_multi_env_1')
    e1 = create_dd_env(1, 'test_multi_env_1')
    basetime = time.time()
    for outeri in range(loops):
        for envi, env in enumerate([e1, e2]):
            if env is None: continue
            env.reset()
            tot_reward = 0.0
            for stepi in range(100):
                angle = stepi * np.pi * 2.0 / 100.0
                x = 512 + np.round(np.cos(angle)*400)
                y = 384 + np.round(np.sin(angle)*300)
                action = [vnc_event.PointerEvent(x, y, 0)]
                obs, reward, done, info = env.step(action)
                obs_sum = np.sum(obs)
                tot_reward += reward
                logger.debug("%d/%d/%d: state=%s sum %.0f, reward=%g, done=%s, action=%s", outeri, envi, stepi, obs.shape, obs_sum, reward, done, action)
                assert obs.shape == (768, 1024, 3)
                assert obs_sum >= 300000000 and obs_sum < 400000000
            logger.info("%d/%d: tot_reward=%.0f", outeri, envi, tot_reward)
            assert tot_reward >= 0
    if e1: e1.close()
    if e2: e2.close()
