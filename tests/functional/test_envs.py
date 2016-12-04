import logging
import os
import pytest
import re

import gym
from universe import wrappers
from universe.runtimes import registration

logger = logging.getLogger(__name__)

# Choose a sample from each category
# TODO: Add more comprehensive test that runs all envs
test_envs = [
    # 'gym-core.PongShortSync-v3',
    # 'gym-core.CartPoleLowDSync-v0',
    'flashgames.DuskDrive-v0',
    'internet.SlitherIO-v0',
    # 'wob.DragBox-v0',
]

@pytest.mark.parametrize('env_id', test_envs)
def test_smoke(env_id):
    """Check that environments start up without errors and that we can extract rewards and observations"""
    gym.undo_logger_setup()
    logging.getLogger().setLevel(logging.INFO)

    env = gym.make(env_id)
    env = wrappers.Unvectorize(env)

    if os.environ.get('FORCE_LATEST_UNIVERSE_DOCKER_RUNTIMES'):  # Used to test universe-envs in CI
        configure_with_latest_docker_runtime_tag(env)
    else:
        env.configure(remotes=1)

    env.reset()
    _rollout(env, timestep_limit=60*30) # Check a rollout

def _rollout(env, timestep_limit=None):
    """
    Test that a rollout follows our desired format. Includes the following checks:

    1. The environment resets and provides an observation within our timestep_limit
    2. Done signals map to the following:

        done=True => Episode over (sent once at end of episode)
        done=None => Resetting, agent takes no actions until done=False again
        done=False => Episode is running, agent should take actions
    """
    count = 0
    episode_state = "resetting"

    while True:
        obs, reward, done, info = env.step([])  # Step with noop action
        count += 1

        if episode_state == 'resetting':
            if done is None:  # Still resetting
                assert obs is None
                continue
            elif done is False:
                episode_state = 'running'

        if episode_state == 'running':
            assert done is False
            assert isinstance(reward, float)
            assert isinstance(done, bool), "Received done=None before done=True"
            # TODO: Remove this None check after we fix done=None semantics
            if obs is not None:
                assert obs['vision'].shape == (768, 1024, 3)
            break

        if timestep_limit is not None and count >= timestep_limit:
            assert episode_state == 'running', "Failed to finish resetting in timestep limit"
            break

        # if timestep_limit is not None and count >= timestep_limit:
        #     self.assertTrue(completed_full_episode, "Failed to complete a full episode in timestep limit")
        #     break

def configure_with_latest_docker_runtime_tag(env):
    original_image = registration.runtime_spec(env.spec.tags['runtime']).image
    latest_image = re.sub(r':.*', ':latest', original_image)
    logger.info("Using latest image: {}".format(latest_image))
    env.configure(remotes=1, docker_image=latest_image)
