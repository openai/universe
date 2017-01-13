import itertools
import json
import logging
import os
import random
import time
from glob import glob

from gym import error
from universe.readers import frame_readers

logger = logging.getLogger(__name__)


def VNCMultiReader(logfile_dirs, *args, **kwargs):
    readers = [VNCReader(logfile_dir, *args, **kwargs) for logfile_dir in logfile_dirs]
    return itertools.chain(*readers)

class VNCReader():
    """
    Iterator that replays a VNCDemonstration at a certain set FPS.
    """
    def __init__(self, logfile_dir,
                 fps=30,
                 metadata_file='metadata.json',
                 reward_file='rewards.demo',
                 observation_file='server.fbs',
                 action_file='client.fbs',
                 botaction_file='botactions.jsonl',
                 crop_to_episode=True,
                 paint_cursor=False,
                 disable_sticky_done_signal=False,
                 ):

        self.logfile_dir = logfile_dir
        self.metadata = self._load_metadata(metadata_file)
        self.start_at, self.end_at = self._get_crop_boundaries_from_metadata(crop_to_episode)

        # Instantiate our frame readers
        if observation_file == 'observations.mp4':
            try:
                video_recording_started_at = self.metadata['recording_started_at']
            except KeyError:
                raise error.Error("MP4 playback requires a metadata.json file with 'recording_started_at' in order to synchronize frames with rewards")
            self.video_reader = frame_readers.FramedVideoReader(self.logfile_dir, video_recording_started_at, start_at=self.start_at, fps=fps)
        else:
            self.video_reader = None

        self.framed_event_reader = frame_readers.FramedEventReader(
            self.logfile_dir,
            fps=fps,
            reward_file=reward_file,
            action_file=action_file,
            botaction_file=botaction_file,
            observation_file=observation_file if observation_file != 'observations.mp4' else None,
            start_at=self.start_at,
            disable_sticky_done_signal=disable_sticky_done_signal,
        )

        # Print a log message every 5 seconds
        self._diagnostics = Diagnostics(print_interval_in_seconds=5.0)
        logger.info("Starting playback from {}".format(self.logfile_dir))

    def _get_crop_boundaries_from_metadata(self, crop_to_episode):
        if crop_to_episode and not self.metadata:
            raise error.Error("Cannot crop to metadata when no metadata file is specified")
        if not self.metadata:
            return None, None

        if crop_to_episode:
            if len(self.metadata['episodes']) > 1:
                logger.warn("Multiple episodes in this demonstration. Cropping to the best episode")
            best_episode = self.metadata['best_episode']
            return float(best_episode['start_time']), float(best_episode['end_time'])
        else:
            return float(self.metadata['recording_started_at']), float(self.metadata['recording_ended_at'])

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        observation, reward, done, info, action = next(self.framed_event_reader)

        if self.video_reader:
            observation = next(self.video_reader)

        self._diagnostics.step()
        return observation, reward, done, info, action

    def _load_metadata(self, metadata_file):
        """
        Args:
            metadata_file: The type of metadata file to look for

        Returns: A dictionary containing the metadata from the file
        """
        if not metadata_file:
            return None

        if metadata_file == '.vexpect.metadata.json':
            logger.warn('suffix .vexpect.metadata.json is considered deprecated')
            metadata_path = self.logfile_dir + '.vexpect.metadata.json'
        else:
            metadata_path = os.path.join(self.logfile_dir, metadata_file)

        if not os.path.isfile(metadata_path):
            logger.warn("No metadata file found at {}. Either generate the metadata file with bin/enrich or set metadata_file=None".format(metadata_path))
            return None

        with open(metadata_path) as f:
            metadata = json.loads(f.readline().strip())  # Only read one line to allow pulling from actions.jsonl

        # Validate metadata
        for required_key in ['recording_started_at']:
            if required_key not in metadata:
                raise error.Error("Bad metadata file: {}. Missing required entry: {}".format(metadata_path, required_key))
        return metadata

class Diagnostics(object):
    def __init__(self, print_interval_in_seconds):
        self._print_interval = print_interval_in_seconds
        self._current_frame = 0
        self._last_diagnostics = {}
        self._last_diagnostics_at = time.time()


    def step(self):
        self._current_frame += 1

        # Check if we should print
        now = time.time()
        delta = now - self._last_diagnostics_at
        if delta > self._print_interval:
            self._print_diagnostics(delta)
            self._last_diagnostics_at = now


    def _print_diagnostics(self, delta):
            frame_delta = self._current_frame - self._last_diagnostics.get("frame", 0)
            diagnostics = {
                'frame': self._current_frame,
                'wallclock_fps': frame_delta / delta
            }
            info = ['{}={}'.format(k, v) for k, v in diagnostics.items()]
            logger.info('Stats for the past %.2fs: %s', delta, ' '.join(info))

            self._last_diagnostics = diagnostics



class VNCDiskReader(object):
    """
    Iterator that replays a VNC demonstration, reading it from disk each time

    Outputs a tuple of the following for each frame:

        action, observation, reward, done, info
    """
    def __init__(self,
                 logfile_dirs,
                 randomize_samples=True,
                 randomize_files=True,
                 queue_size=25,
                 max_samples=-1,
                 *args,
                 **kwargs):
        self.logfile_dirs = logfile_dirs
        self.args = args
        self.kwargs = kwargs
        self.randomize_samples = randomize_samples
        self.randomize_files = randomize_files
        self.queue_size = queue_size
        self.max_samples = max_samples
        self.counter = 0
        self.epoch = -1
        self.reader = []
        self._start_new_epoch()
        self.queue = []

    def __iter__(self):
        return self

    def _start_new_epoch(self):
        if self.randomize_files:
            random.shuffle(self.logfile_dirs)
        self.reader = VNCMultiReader(self.logfile_dirs, *self.args, **self.kwargs)
        self.counter = 0
        self.epoch += 1

    def next(self):
        return self.__next__()

    def __next__(self):
        while len(self.queue) < self.queue_size:
            if self.counter == self.max_samples:
                self._start_new_epoch()
            try:
                observation, reward, done, info, action = next(self.reader)
            except StopIteration:
                self._start_new_epoch()
                observation, reward, done, info, action = next(self.reader)
            self.queue.append((observation, reward, done, info, action))
            self.counter += 1
        n_pop = random.randint(0, len(self.queue)) if self.randomize_samples else 0
        return _pop_queue(self.queue, n_pop)


def _pop_queue(queue, index):
    pop_value = queue[index]
    queue[index] = queue[-1]
    queue.pop()
    return pop_value

class ReaderFilter(object):
    def __init__(self, reader):
        self.reader = reader

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        raise NotImplementedError

class ObservationFilter(ReaderFilter):
    def __init__(self, reader, obs_filter):
        super(ObservationFilter, self).__init__(reader)
        self.obs_filter = obs_filter

    def next(self):
        return self.__next__()

    def __next__(self):
        observation, reward, done, info, action = next(self.reader)
        return self.obs_filter(observation), reward, done, info, action

class ActionFilter(ReaderFilter):
    def __init__(self, reader, action_filter):
        super(ActionFilter, self).__init__(reader)
        self.action_filter = action_filter

    def next(self):
        return self.__next__()

    def __next__(self):
        observation, reward, done, info, action = next(self.reader)
        return observation, reward, done, info, self.action_filter(action)


class MemoryReader(ReaderFilter):
    """
    Iterator that replays a series of VNC demos, reading them from disk the first
    time and then storing them in memory and looping thereafter.
    Outputs a tuple of the following for each frame:

        action, observation, reward, done, info
    """
    def __init__(self, reader, randomize_samples=True, max_samples=-1):
        super(MemoryReader, self).__init__(reader)
        self.max_samples = max_samples
        self.randomize_samples = randomize_samples
        self.epoch = 0
        self.counter = 0
        self.memory = []

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.counter == self.max_samples:
            self.counter = 0
            self.epoch += 1
            if self.randomize_samples:
                random.shuffle(self.memory)
        if self.epoch == 0:
            try:
                observation, reward, done, info, action = next(self.reader)
            except StopIteration:
                self.max_samples = self.counter
                if len(self.memory) == 0:
                    raise Exception("Could not read any samples from this demo.")
                return self.memory[0]
            self.memory.append((observation, reward, done, info, action))
        self.counter += 1
        return self.memory[self.counter-1]

class RewardFilteringReader(ReaderFilter):
    def __init__(self, reader, reward_threshold):
        super(RewardFilteringReader, self).__init__(reader)
        self.reward_threshold = reward_threshold
        self.count = 0
        self.episode = 0
        self.total_reward = 0
        self.total_samples = 0
        self.episode_buffer = []

    def next(self):
        return self.__next__()

    def __next__(self):
        self.count += 1
        if self.count >= len(self.episode_buffer):
            self._get_new_episode()
        return self.episode_buffer[self.count]

    def _get_new_episode(self):
        first = True
        self.count = 0
        # This could in theory infinite loop...
        r = 0.
        while first or r < self.reward_threshold:
            first = False
            r = 0.
            self.episode_buffer = []
            done = False
            while not done:
                observation, reward, done, info, action = next(self.reader)
                r += reward
                self.episode_buffer.append((observation, reward, done, info, action))
        self.total_reward += r
        self.episode += 1
        self.total_samples += len(self.episode_buffer)
        print("found reward", r)
        print("average reward so far", float(self.total_reward)/float(self.episode))
        print("episodes so far", self.episode)
        print("samples so far", self.total_samples)


def read_batch(reader, batch_size):
    observations, rewards, dones, infos, actions = zip(*[next(reader) for _ in range(batch_size)])
    return list(observations), list(rewards), list(dones), list(infos), list(actions)


def get_demo_files(download_dir, env_id):
    demo_dir = os.path.join(download_dir, env_id)
    return get_demo_files_dir(demo_dir)


def get_demo_files_dir(demo_dir):
    demo_files = glob(os.path.join(demo_dir, "*","*"))
    return [d for d in demo_files if os.path.isdir(d)]
