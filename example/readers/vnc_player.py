#!/usr/bin/env python3
import argparse
import logging
import sys
import time

from universe.readers.vnc_reader import VNCMultiReader
import numpy as np
import six

logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-d', '--logfile-dirs', required=True, nargs='+', help='Directories with client.fbs, server.fbs, and rewards.demo, (e.g: -d /tmp/demos/VNCSpaceInvaders-v0/*/*  ')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-R', '--no-render', action='store_true', help='Do not render.')
    parser.add_argument('-o', '--omit-reward', action='store_true', default=False, help='Do not look for a rewards.demo file, just replay the .fbs files')
    parser.add_argument('-s', '--sanity-check', action='store_true', help='Just sanity check the demo: show the first and last frame')
    parser.add_argument('-m', '--metadata-file', default='metadata.json', help='Crop using metadata information, e.g rewards.metadata.json, metadata.json')
    parser.add_argument('-r', '--reward-file', default='rewards.jsonl', help='rewards file to use from the logfile dirs, e.g rewards.jsonl, rewards.demo')
    parser.add_argument('-O', '--observation-file', default='server.fbs', help='observation file to use from the logfile dirs, e.g server.fbs, observations.mp4')
    parser.add_argument('-B', '--botaction-file', default='botactions.jsonl', help='bot actions file to use from the logfile dirs, e.g botactions.jsonl')
    parser.add_argument('-f', '--fps', default=30., help='Assume this many frames per second (higher values will look slower)')
    parser.add_argument('-S', '--no-sticky-done-signal', action='store_true', help='Disable sticky done signal')
    parser.add_argument('-C', '--no-crop-to-episode', action='store_true', help='Do not crop to episode')

    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    if args.omit_reward:
        reward_file = None
    else:
        reward_file = args.reward_file

    reader = VNCMultiReader(args.logfile_dirs,
                            metadata_file=args.metadata_file,
                            reward_file=reward_file,
                            observation_file=args.observation_file,
                            botaction_file=args.botaction_file,
                            fps=float(args.fps),
                            crop_to_episode=not args.no_crop_to_episode,
                            disable_sticky_done_signal=args.no_sticky_done_signal,
                            )
    viewer = None
    if args.sanity_check or not args.no_render:
        from gym.envs.classic_control import rendering
        viewer = rendering.SimpleImageViewer()

    start_timestamp = None
    displayed = False
    for i, (observation, reward, done, info, action) in enumerate(reader):
        if observation is None:
            continue

        if action != []:
            logger.info('Action (VNCEvents): {}'.format(action))
        if reward != 0.:
            logger.info('Reward: {}'.format(reward))
        if done:
            logger.info('Done: {}'.format(done))
        if info != []:
            logger.debug('Info: {}'.format(info))

        timestamp = info.get('reader.timestamp', None)
        if not start_timestamp:
            start_timestamp = timestamp

        # Print everything
        logger.debug("""
        action: {}
        observation: {}
        reward: {}
        done: {}
        info: {}
        """.format(action, len(observation), reward, done, info ))

        # Render observation
        if args.sanity_check:
            if not displayed and not np.all(observation == 0):
                displayed = True
                viewer.imshow(observation)
        elif not args.no_render:
            viewer.imshow(observation)
            time.sleep(1. / 60.)

    # Sanity check the last frame
    if args.sanity_check:
        viewer.imshow(observation)
        if six.PY2:
            pause = raw_input
        else:
            pause = input
        pause("press enter to terminate")

    return 0

if __name__ == '__main__':
    sys.exit(main())
