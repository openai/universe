import json
import logging

import universe
import imageio
from universe.readers import playback

logger = logging.getLogger(__name__)


class FramedEventReader(object):
    """
    Iterator that reads in event files and returns aggregated values at a set FPS
    """
    def __init__(self, logfile_dir, fps=30, reward_file='rewards.demo', observation_file='server.fbs', start_at=None, paint_cursor=False, action_file='client.fbs', botaction_file='botactions.jsonl', progress=False, disable_sticky_done_signal=False):
        assert observation_file is None or observation_file.endswith('.fbs'), "No such evented observation file type: {}".format(observation_file)
        self.playback = playback.Playback(
            logfile_dir=logfile_dir,
            observation_file=observation_file,
            reward_file=reward_file,
            paint_cursor=paint_cursor,
            action_file=action_file,
            botaction_file=botaction_file
        )
        self._frame_length_in_s = 1.0 / fps

        self.observation_for_frame = None
        self._last_playback_event = None
        self._progress = progress
        self._scanned = 0.

        self._disable_sticky_done_state = disable_sticky_done_signal
        if disable_sticky_done_signal:
            logger.warn("Warning: Disabling sticky done signal. This is a legacy setting for World of Bits environments that will set any frame that does not have an explicit `done=True` value to `done=False`. Please adjust the environment that is writing these files to return an explicit `done=False` after it finishes resetting.")

        self._last_done_signal = None
        self._recieved_true_done_signal_in_frame = False

        # Take the first event and then advance to the start
        self._next_playback_event = self.playback.next()

        self._advance_playback_until_start(start_at)

    def _advance_playback_until_start(self, start_at):
        """
        Seek to the start of the demonstration, applying actions as we go.
        The first frame that we return will have all the actions that we've seen since the beginning of Playback
        """
        self._reset_frame()
        try:
            if start_at:
                # Set initial playback event in case we don't have any actions within this frame
                self._last_playback_event = {'timestamp': start_at}
                self._advance_playback_until_timestamp(start_at)

                # Discard the aggregated infos and rewards
                self.info_for_frame = {}
                self.reward_for_frame = 0.
            else:
                self._advance_playback()
        except StopIteration:
            logger.error("Bad demo, reached the end without getting an observation: {}".format(self.playback))

        self.end_of_frame_timestamp = self._last_playback_event['timestamp']

    def _advance_playback(self):
        """Apply the previously queued playback event and queue up the next one"""
        # Apply the previously cached event if we have one
        if self._next_playback_event:
            self._last_playback_event = self._next_playback_event
            self._apply_playback_event_to_frame(self._next_playback_event)

        # Then queue up the next one
        self._next_playback_event = self.playback.next()

    def _advance_playback_until_timestamp(self, timestamp):
        while self._next_playback_event['timestamp'] < timestamp:
            self._advance_playback()

    def _apply_playback_event_to_frame(self, playback_event):
        """Applies the playback_event to the current frame"""

        observation = playback_event.get('observation')
        if observation is not None:
            self.observation_for_frame = observation  # Keep the same observation if we didn't get a new one.

        self.action_for_frame += playback_event.get('action', [])
        self.reward_for_frame += playback_event.get('reward', 0.)

        if 'done' in playback_event:
            done = playback_event['done']
            if done is True:
                self._recieved_true_done_signal_in_frame = True  # Mark this frame, we will return True at least once
            self._last_done_signal = done

        universe.merge_infos(self.info_for_frame, playback_event.get('info', None) or {})
        return playback_event['timestamp']

    def _reset_frame(self):
        """Reset everything from the last frame. Keep the observation and done, which persist until changed"""
        self.action_for_frame = []
        self.reward_for_frame = 0.
        self.info_for_frame = {}

        self._recieved_true_done_signal_in_frame = False

    def __iter__(self):
        return self

    def next(self):
        """
        Returns a 5-tuple of (observation, reward, done, info, actions) where:
        - observation is (768,1024,3) uint8 numpy array with pixels 0..255.
          Important Note: the observation returned from this function is always
          the same numpy array but its contents change. This is done for runtime
          efficiency to prevent always allocating a new numpy array.
        - reward is a float
        - done is a bool
        - info is a dict of auxiliary information (e.g. timestamp)
        - actions is a list of VNC events (universe.vnc_spaces.[PointerEvent|KeyEvent])
          that occured in the span of 1/self.fps of time.
        """
        return self.__next__()

    def __next__(self):
        self.info_for_frame['reader.frame_start_at'] = self.end_of_frame_timestamp
        self.end_of_frame_timestamp += self._frame_length_in_s

        # Read while end_of_frame is still further ahead than the next event
        self._advance_playback_until_timestamp(self.end_of_frame_timestamp)

        self.info_for_frame['reader.last_playback_event_at'] = self._last_playback_event['timestamp']
        self.info_for_frame['reader.next_playback_event_at'] = self._next_playback_event['timestamp']
        self.info_for_frame['reader.frame_end_at'] = self.end_of_frame_timestamp

        if self._disable_sticky_done_state:
            done = self._recieved_true_done_signal_in_frame
        else:
            done = True if self._recieved_true_done_signal_in_frame else self._last_done_signal

        output = (
            self.observation_for_frame,
            self.reward_for_frame,
            done,
            self.info_for_frame,
            self.action_for_frame
        )

        # Reset at the end, rather than the beginning, because we don't want to clobber the
        # aggregated state from _apply_playback_until_start on the first frame.
        self._reset_frame()

        self._scanned += self._frame_length_in_s
        if self._progress:
            print('scanning progress: {:0.2f} seconds'.format(self._scanned), end='\r')

        return output


class FramedVideoReader(object):
    def __init__(self, logfile_dir, video_started_at, start_at, fps=30.):
        self._reader = imageio.get_reader(os.path.join(logfile_dir, 'observations.mp4'), 'ffmpeg')
        assert self._reader.get_meta_data()['fps'] == fps, "FramedVideoReader was given a framerate that doesn't match mp4"
        self.fps = fps

        self._playback_head = video_started_at
        while self._playback_head < start_at:
            next(self)

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __next__(self):
        self._playback_head += 1./self.fps
        try:
            return self._reader.get_next_data()
        except IndexError:
            raise StopIteration
