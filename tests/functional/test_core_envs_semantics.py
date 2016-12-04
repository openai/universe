import logging
import pytest

import gym
import numpy as np
from PIL import Image
from gym import spaces
from universe import wrappers
from universe.envs.vnc_core_env import translator

def show(obs):
    Image.fromarray(obs).show()

class AtariMatcher(object):
    def translator(self, env):
        return translator.AtariTranslator(env)

    def crop(self, obs):
        return obs[20:210, :160, :]

    def assert_match(self, obs, vnc_obs, extra_info=None, stage=None):
        # Crop out the mouse
        vnc_obs_cropped = self.crop(vnc_obs)
        obs_cropped = self.crop(obs)

        if not np.all(vnc_obs_cropped == obs_cropped):
            show(vnc_obs_cropped)
            show(obs_cropped)
            show(vnc_obs_cropped - obs_cropped)
            assert False, '[{}] Observations do not match: vnc_obs_cropped={} obs_cropped={} extra_info={}'.format(stage, vnc_obs_cropped, obs_cropped, extra_info)

# Wraps an Atari-over-VNC env so that it behaves like a vectorized vanilla Atari env
def atari_vnc_wrapper(env):
    env = wrappers.Vision(env)
    env = wrappers.GymCoreAction(env)
    return env

class CartPoleLowDMatcher(object):
    def translator(self, env):
        return translator.CartPoleTranslator(env)

    def assert_match(self, obs, vnc_obs, extra_info=None, stage=None):
        assert np.all(np.isclose(obs, vnc_obs)), '[{}] Observations do not match: vnc_obs={} obs={}'.format(stage, vnc_obs, obs)

def reset(matcher, env, vnc_env, stage=None):
    obs = env.reset()
    vnc_obs = vnc_env.reset()
    matcher.assert_match(obs, vnc_obs, stage=stage)

def rollout(matcher, env, vnc_env, timestep_limit=None, stage=None):
    count = 0
    actions = matcher.translator(env)

    done = None
    while True:
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        if done:
            # Account for remote auto-reset
            obs = env.reset()

        vnc_obs, vnc_reward, vnc_done, vnc_info = vnc_env.step(action)
        assert reward == vnc_reward
        assert done == vnc_done
        assert vnc_info['stats.reward.count'] == 1
        matcher.assert_match(obs, vnc_obs, {'reward': reward, 'done': done}, stage=stage)

        count += 1
        if done or (timestep_limit is not None and count >= timestep_limit):
            break

# TODO: we should have auto-env spinup
specs = [
    (gym.spec('gym-core.PongDeterministicSync-v3'), AtariMatcher(), atari_vnc_wrapper),
    (gym.spec('gym-core.PitfallDeterministicSync-v3'), AtariMatcher(), atari_vnc_wrapper),

    # This test is still broken. Looks like we're not piping the seed
    # to the CartPole env behind VNC
#    (gym.spec('gym-core.CartPoleLowDSync-v0'), CartPoleLowDMatcher())
]

@pytest.mark.parametrize("spec,matcher,wrapper", specs)
def test_nice_vnc_semantics_match(spec, matcher, wrapper):
    # Check that when running over VNC or using the raw environment,
    # semantics match exactly.
    gym.undo_logger_setup()
    logging.getLogger().setLevel(logging.INFO)

    spaces.seed(0)

    vnc_env = spec.make()
    vnc_env = wrapper(vnc_env)
    vnc_env = wrappers.Unvectorize(vnc_env)
    vnc_env.configure(remotes=1)

    env = gym.make(spec._kwargs['gym_core_id'])

    env.seed(0)
    vnc_env.seed(0)

    # Check that reset observations work
    reset(matcher, env, vnc_env, stage='initial reset')

    # Check a full rollout
    rollout(matcher, env, vnc_env, timestep_limit=50, stage='50 steps')

    # Reset to start a new episode
    reset(matcher, env, vnc_env, stage='reset to new episode')

    # Check that a step into the next episode works
    rollout(matcher, env, vnc_env, timestep_limit=1, stage='1 step in new episode')

    # Make sure env can be reseeded
    env.seed(1)
    vnc_env.seed(1)
    reset(matcher, env, vnc_env, 'reseeded reset')
    rollout(matcher, env, vnc_env, timestep_limit=1, stage='reseeded step')
