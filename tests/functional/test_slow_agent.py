import logging
import os, time, re
import pytest
import numpy as np
from six.moves import queue
import gym, universe
from universe import wrappers, spaces
from universe.vncdriver import constants
from universe.spaces import vnc_event
import universe.wrappers.logger
import logging.handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

def run_slow_agent(agent_think_time):
    """
    This tests with a slow agent, that takes > 150 mS to think even though we asked for 10 fps from the env.
    We should get observations like normal, just at larger intervals.
    Typical results:
    Slow agent: reaction_time_samples=[426.99, 387.44, 472.76, 473.69, 439.77] fps_samples=[9.84, 9.85, 9.85, 9.85, 9.87, 9.87, 9.89, 7.89, 5.94, 5.93, 6.11, 5.86, 5.8, 5.47]
    """

    errors = []

    mhq = queue.Queue()
    universe.wrappers.logger.logger.addHandler(logging.handlers.QueueHandler(mhq))

    # Figure out a matrix that takes about 100 mS to invert
    bigmat_size = 32
    while True:
        bigmat = np.random.rand(bigmat_size, bigmat_size)
        t0 = time.time()
        np.linalg.inv(bigmat)
        t1 = time.time()
        if 0: logger.debug('bigmat_size=%d t=%0.3f', bigmat_size, t1-t0)
        if t1-t0 >= agent_think_time:
            break
        bigmat_size = int(bigmat_size*1.0625)
    logger.debug('bigmat_size=%d t=%0.3f', bigmat_size, t1-t0)


    env = create_dd_env(1, 'test_slow_agent')
    basetime = time.time()
    env.reset()
    tot_reward = 0.0
    prev_obs = None
    dup_obs_count = 0
    for stepi in range(200):
        action = [spaces.KeyEvent.by_name('up', down=(stepi%6<3))]

        t0 = time.time()
        obs, reward, done, info = env.step(action)
        # obs = obs[84:XXX, 18:XXX, :] * (1.0/255.0)

        t1 = time.time()
        np.linalg.inv(bigmat)
        obs_sum = np.sum(obs)
        tot_reward += reward
        t2 = time.time()

        obs_diff = np.sum(np.abs(obs - prev_obs)) if (prev_obs is not None) else 0
        if obs_diff == 0:
            dup_obs_count += 1
        prev_obs = np.copy(obs)

        logger.debug("slow_agent times step=%0.3f agent=%0.3f | action=%s state=%s sum %0.f diff %.0f reward=%0.3f done=%s", t1-t0, t2-t1, action, obs.shape, obs_sum, obs_diff, reward, done)
        assert obs.shape == (768, 1024, 3)
        if not (obs_sum >= 300000000 and obs_sum < 400000000):
            errors.append('Invalid obs_sum=%.0f' % obs_sum)

    logger.info("tot_reward=%.0f dup_obs_count=%d", tot_reward, dup_obs_count)

    reaction_time_samples = []
    fps_samples = []
    while True:
        try:
            item = mhq.get_nowait()
        except queue.Empty:
            break
        logger.debug('Captured: %s', item.message)
        m1 = re.search(' reaction_time=([\d\.]+)ms', item.message)
        if m1:
            reaction_time_samples.append(float(m1.group(1)))
        m1 = re.search(' fps=([\d\.]+)', item.message)
        if m1:
            fps_samples.append(float(m1.group(1)))

    env.close()

    return reaction_time_samples, fps_samples, tot_reward, dup_obs_count, errors


def test_slow_agent():
    """
    Tests an agent that runs slow -- we ask for 10 fps from the env, but then take more than 100 mS to think
    so the frame rate drops. We want to be sure it handles this smoothly.
    """

    reaction_time_samples, fps_samples, tot_reward, dup_obs_count, errors = run_slow_agent(0.15)

    if not (tot_reward >= 8000 and tot_reward < 20000):
        errors.append('Invalid tot_reward=%.0f' % tot_reward)
    if not (dup_obs_count < 3):
        errors.append('Invalid dup_obs_count=%d' % dup_obs_count)

    logger.info('Slow agent: reaction_time_samples=%s fps_samples=%s', reaction_time_samples, fps_samples)
    if len(reaction_time_samples) < 2:
        errors.append('Not enough reaction time samples')
    else:
        reaction_time_avg = np.mean(reaction_time_samples)
        if reaction_time_avg < 100.0 or reaction_time_avg > 600.0:
            errors.append('Bad reaction_time_avg=%s' % reaction_time_avg)

    if len(fps_samples) < 2:
        errors.append('Not enough fps samples')
    else:
        fps_avg = np.mean(fps_samples)
        if fps_avg < 5 or fps_avg > 11:
            errors.append('Bad fps_avg=%s' % fps_avg)
        fps_min = np.min(fps_samples)
        if fps_min < 4:
            errors.append('Bad fps_min=%s' % fps_min)

    if errors: raise ValueError(errors)



def test_fast_agent():
    """
    Tests an agent that runs fast -- we ask for 10 fps from the env and should get it.
    Typical results:
      Fast agent: reaction_time_samples=[168.23, 196.7] fps_samples=[9.85, 9.87, 9.82, 9.87, 9.84, 9.84, 9.88, 9.86, 9.87, 9.86, 8.37, 9.83, 9.81]

    """

    reaction_time_samples, fps_samples, tot_reward, dup_obs_count, errors = run_slow_agent(0.01)

    if not (tot_reward >= 8000 and tot_reward < 20000):
        errors.append('Invalid tot_reward=%.0f' % tot_reward)
    if not (dup_obs_count < 10):
        errors.append('Invalid dup_obs_count=%d' % dup_obs_count)

    logger.info('Fast agent: reaction_time_samples=%s fps_samples=%s', reaction_time_samples, fps_samples)
    if len(reaction_time_samples) < 2:
        errors.append('Not enough reaction time samples')
    else:
        reaction_time_avg = np.mean(reaction_time_samples)
        if reaction_time_avg < 10.0 or reaction_time_avg > 200.0:
            errors.append('Bad reaction_time_avg=%s' % reaction_time_avg)

    if len(fps_samples) < 2:
        errors.append('Not enough fps samples')
    else:
        fps_avg = np.mean(fps_samples)
        if fps_avg < 8 or fps_avg > 11:
            errors.append('Bad fps_avg=%s' % fps_avg)
        fps_min = np.min(fps_samples)
        if fps_min < 7:
            errors.append('Bad fps_min=%s' % fps_min)


    if errors: raise ValueError(errors)
