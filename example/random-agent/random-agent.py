#!/usr/bin/env python
import argparse
import logging
import sys

import gym
import universe # register the universe environments

from universe import wrappers

logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)


    env = gym.make('flashgames.NeonRace-v0')
    env.configure(remotes=1)  # automatically creates a local docker container
    
    # Restrict the valid random actions. (Try removing this and see
    # what happens when the agent is given full control of the
    # keyboard/mouse.)
    env = wrappers.experimental.SafeActionSpace(env)
    observation_n = env.reset()

    while True:
        # your agent here
        #
        # Try sending this instead of a random action: ('KeyEvent', 'ArrowUp', True)
        action_n = [env.action_space.sample() for ob in observation_n]
        observation_n, reward_n, done_n, info = env.step(action_n)
        env.render()

    return 0

if __name__ == '__main__':
    sys.exit(main())
