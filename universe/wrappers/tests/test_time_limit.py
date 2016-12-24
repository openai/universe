import gym
import time
import universe
from gym.envs import register
from universe import wrappers

register(
    id='test.SecondsLimitDummyVNCEnv-v0',
    entry_point='universe.envs:DummyVNCEnv',
    tags={
        'vnc': True,
        'wrapper_config.TimeLimit.max_episode_seconds': 0.1
        }
    )

register(
    id='test.StepsLimitDummyVNCEnv-v0',
    entry_point='universe.envs:DummyVNCEnv',
    tags={
        'vnc': True,
        'wrapper_config.TimeLimit.max_episode_steps': 1
        }
    )


def test_steps_limit_restart():
    env = gym.make('test.StepsLimitDummyVNCEnv-v0')
    env = wrappers.TimeLimit(env)
    env.configure(_n=1)

    assert env.max_episode_seconds == None
    assert env.max_episode_steps == 1

    # Episode has started
    _, _, done, info = env.step([[]])
    assert done == [False]

    # Limit reached, now we get a done signal and the env resets itself
    _, _, done, info = env.step([[]])
    assert done == [True]


def test_steps_limit_restart_unused_when_not_wrapped():
    env = gym.make('test.StepsLimitDummyVNCEnv-v0')
    env.configure(_n=1)

    for i in range(10):
        _, _, done, info = env.step([[]])
        assert done == [False]


def test_seconds_limit_restart():
    env = gym.make('test.SecondsLimitDummyVNCEnv-v0')
    env = wrappers.TimeLimit(env)
    env.configure(_n=1)

    assert env.max_episode_seconds == 0.1
    assert env.max_episode_seconds == None

    # Episode has started
    _, _, done, info = env.step([[]])
    assert done == [False]

    # Not enough time has passed
    _, _, done, info = env.step([[]])
    assert done == [False]

    time.sleep(0.2)

    # Limit reached, now we get a done signal and the env resets itself
    _, _, done, info = env.step([[]])
    assert done == [True]


def test_default_time_limit():
    env = gym.make('test.DummyVNCEnv-v0')
    env = wrappers.TimeLimit(env)
    env.configure(_n=1)

    assert env.max_episode_seconds == wrappers.time_limit.DEFAULT_MAX_EPISODE_SECONDS
    assert env.max_episode_steps == None

