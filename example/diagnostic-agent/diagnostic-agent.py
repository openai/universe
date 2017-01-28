#!/usr/bin/env python
import argparse
import logging
import time

import gym
import numpy as np
import universe
from universe import pyprofile, wrappers, spaces
from gym import wrappers as gym_wrappers

# if not os.getenv("PYPROFILE_FREQUENCY"):
#     pyprofile.profile.print_frequency = 5
from universe import vectorized

logger = logging.getLogger()

CHROME_X_OFFSET = 18
CHROME_Y_OFFSET = 84

class NoopSpace(gym.Space):
    """ Null action space """
    def sample(self, seed=0):
        return []
    def contains(self, x):
        return x == []

class ForwardSpace(gym.Space):
    """ Only move forward action space """
    def __init__(self, key='w'):
        self.key = [spaces.KeyEvent.by_name(key, down=True)]
    def sample(self, seed=0):
        return self.key
    def contains(self, x):
        return x == self.key

# The world's simplest agent!
class RandomAgent(object):
    """
    Example usage:

        bin/random_agent.py -e gym-core.Pong-v3 --remote localhost:5900+15900

    """
    def __init__(self, action_space, n, vectorized):
        self.action_space = action_space
        self.n = n
        self.vectorized = vectorized

    def __call__(self, observation, reward, done):
        if self.vectorized:
            return [self.action_space.sample() for _ in range(self.n)]
        else:
            return self.action_space.sample()

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger.setLevel(logging.INFO)
    universe.configure_logging()

    # Actions this agent will take, 'random' is the default
    action_choices = ['random', 'noop', 'forward', 'click']

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_id', default='gym-core.Pong-v3', help='Which environment to run on.')
    parser.add_argument('-m', '--monitor', action='store_true', help='Whether to activate the monitor.')
    parser.add_argument('-r', '--remote', default=None, help='The number of environments to create (e.g. -r 20), or the address of pre-existing VNC servers and rewarders to use (e.g. -r vnc://localhost:5900+15900,localhost:5901+15901)')
    parser.add_argument('-c', '--client-id', default='0', help='Set client id.')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-R', '--no-render', action='store_true', help='Do not render the environment locally.')
    parser.add_argument('-A', '--actions', choices=action_choices, default='random', help='How to sample actions to send to remote environment')
    parser.add_argument('-d', '--docker-image', help='Force a version of the docker_image used with --remote <int>. e.g --docker-image quay.io/openai/universe.gym-core:0.3')
    parser.add_argument('-s', '--reuse', default=False, action='store_true', help='Reuse existing Docker container if present, and leave this one running after (only for "-r n")')
    parser.add_argument('-f', '--fps', default=60., type=float, help='Desired frames per second')
    parser.add_argument('-N', '--max-steps', type=int, default=10**7, help='Maximum number of steps to take')
    parser.add_argument('-E', '--max-episodes', type=int, default=10**7, help='Maximum number of episodes')
    parser.add_argument('-T', '--start-timeout', type=int, default=None, help='Rewarder session connection timeout (seconds)')
    args = parser.parse_args()

    logging.getLogger('gym').setLevel(logging.NOTSET)
    logging.getLogger('universe').setLevel(logging.NOTSET)
    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    if args.env_id is not None:
        env = gym.make(args.env_id)
    else:
        env = wrappers.WrappedVNCEnv()
    # env = wrappers.BlockingReset(env)
    if not isinstance(env, wrappers.GymCoreAction):
        # The GymCoreSyncEnv's try to mimic their core counterparts,
        # and thus came pre-wrapped wth an action space
        # translator. Everything else probably wants a SafeActionSpace
        # wrapper to shield them from random-agent clicking around
        # everywhere.
        env = wrappers.experimental.SafeActionSpace(env)
    else:
        # Only gym-core are seedable
        env.seed([0])
    env = wrappers.Logger(env)

    if args.monitor:
        env = wrappers.Monitor(env, '/tmp/vnc_random_agent', force=True)

    if args.actions == 'random':
        action_space = env.action_space
    elif args.actions == 'noop':
        action_space = NoopSpace()
    elif args.actions == 'forward':
        action_space = ForwardSpace()
    elif args.actions == 'click':
        spec = universe.runtime_spec('flashgames').server_registry[args.env_id]
        height = spec["height"]
        width = spec["width"]
        noclick_regions = [r['coordinates'] for r in spec['regions'] if r['type'] == 'noclick'] if spec.get('regions') else []
        active_region = (CHROME_X_OFFSET, CHROME_Y_OFFSET, CHROME_X_OFFSET + width, CHROME_Y_OFFSET + height)
        env = wrappers.SoftmaxClickMouse(env, active_region=active_region, noclick_regions=noclick_regions)
        action_space = env.action_space
    else:
        logger.error("Invalid action choice: {}".format(args.actions))
        exit(1)

    env.configure(
        fps=args.fps,
        # print_frequency=None,
        # ignore_clock_skew=True,
        remotes=args.remote,
        client_id=args.client_id,
        start_timeout=args.start_timeout,

        # remotes=remote, docker_image=args.docker_image, reuse=args.reuse, ignore_clock_skew=True,
        # vnc_session_driver='go', vnc_session_kwargs={
        #     'compress_level': 0,
        # },

        vnc_driver='go', vnc_kwargs={
            # 'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50, 'subsample_level': 2,
            'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50, 'subsample_level': 0, 'quality_level': 5,
        },
    )

    agent = RandomAgent(action_space, n=env.n, vectorized=env.metadata['runtime.vectorized'])

    render = not args.no_render
    observation_n = env.reset()
    target = time.time()
    reward_n = [0] * env.n
    done_n = [False] * env.n

    observation_count = np.zeros(env.n)
    episode_length = np.zeros(env.n)
    episode_score = np.zeros(env.n)

    episodes_completed = 0

    for i in range(args.max_steps):
        # print(observation_n)
        # user_input.handle_events()

        if render:
            # Note the first time you call render, it'll be relatively
            # slow and you'll have some aggregated rewards. We could
            # open the render() window before `reset()`, but that's
            # confusing since it pops up a black window for the
            # duration of the reset.
            env.render()

        action_n = agent(observation_n, reward_n, done_n)

        # Take an action
        with pyprofile.push('env.step'):
            observation_n, reward_n, done_n, info = env.step(action_n)

        episode_length += 1
        if not all(r is None for r in reward_n): # checks if we connected the rewarder
            episode_score += np.array(reward_n)
        for i, ob in enumerate(observation_n):
            if ob is not None and (not isinstance(ob, dict) or ob['vision'] is not None):
                observation_count[i] += 1

        scores = {}
        lengths = {}
        observations = {}
        for i, done in enumerate(done_n):
            if not done:
                continue
            scores[i] = episode_score[i]
            lengths[i] = episode_length[i]
            observations[i] = observation_count[i]

            episode_score[i] = 0
            episode_length[i] = 0
            observation_count[i] = 0
        if len(scores) > 0:
            logger.info('Total for completed episodes: reward=%s length=%s observations=%s', scores, lengths, observations)

        errored = [i for i, info_i in enumerate(info['n']) if 'error' in info_i]
        if errored:
            logger.info('had errored indexes: %s: %s', errored, info)

        episodes_completed += len([d for d in done_n if d])
        if episodes_completed >= args.max_episodes:
            break

        # if info.get('n') and info['n'][0].get('env_status.instruction'):
        #     logger.info('received instruction = %s', info['n'][0]['env_status.instruction'])

        # if observation_n[0].get('text'):
        #     logger.info('message_n=%s', [observation['text'] for observation in observation_n])

        # if any(done_n) or any(r != 0.0 and r is not None for r in reward_n):
        #     logger.info('reward_n=%s done_n=%s info=%s', reward_n, done_n, info)

    # We're done! clean up
    env.close()
