#!/usr/bin/env python
"""Parse a server.fbs file and save screenshots every 30 seconds."""

import argparse
import logging
import os
import sys

try:
    from scipy.misc import imsave
except ImportError:
    from imageio import imwrite as imsave

from universe.readers.vnc_reader import VNCReader
import numpy as np

logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('logfile_dir', help='Directory with client.fbs and server.fbs')
    parser.add_argument('-o', '--output-dir', required=True, help='directory in which to write screenshots (created if necessary)')
    parser.add_argument('-O', '--observation-file', default='server.fbs', help='observation file to use from the logfile dirs, e.g server.fbs, observations.mp4')
    parser.add_argument('-i', '--interval', default=30, type=int, help='take a screenshot every `interval` seconds')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    logger.info('mkdir -p {}'.format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    reader = VNCReader(args.logfile_dir,
                       observation_file=args.observation_file,
                       crop_to_episode=False,
                       reward_file=None)

    last_screenshot_time = None
    seconds = 0
    for observation, _, done, info, action in reader:
        if observation is None:
            continue

        if last_screenshot_time is None:
            # skip the first screenshot
            last_screenshot_time = info['reader.frame_start_at']

        if info['reader.frame_start_at'] - last_screenshot_time >= args.interval:
            last_screenshot_time = info['reader.frame_start_at']
            seconds += args.interval
            img_path = os.path.join(args.output_dir, 'screenshot-{}.png'.format(seconds))
            logger.info('writing to {}'.format(img_path))
            imsave(img_path, observation)



if __name__ == '__main__':
    sys.exit(main())
