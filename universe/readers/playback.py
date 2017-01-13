import logging
import os

from universe import error
from universe.readers import event_readers

logger = logging.getLogger(__name__)


class InvalidDemonstrationError(error.Error):
    pass


def Playback(logfile_dir, observation_file=None, action_file=None, reward_file=None, paint_cursor=False, botaction_file=None):
    """
    Iterator that replays a VNCDemonstration event by event
    Outputs a dict of the following:
        { timestamp, action, observation, reward, done, info }
    """
    observation_path = os.path.join(logfile_dir, observation_file) if observation_file else None
    action_path      = os.path.join(logfile_dir, action_file) if action_file else None
    botaction_path   = os.path.join(logfile_dir, botaction_file) if botaction_file else None
    reward_path      = os.path.join(logfile_dir, reward_file) if reward_file else None

    expected_files = [observation_path, action_path, reward_path]
    missing_files = [file for file in expected_files if (file is not None and not os.path.exists(file))]
    if missing_files:
        raise InvalidDemonstrationError("Could not start VNC Playback. Missing required files: {}".format(missing_files))

    readers = []

    if observation_file == 'server.fbs' and action_file != 'client.fbs':
        raise error.Error(
            "Cannot include server.fbs without client.fbs because client.fbs includes pixel format information")

    if observation_file:
        # Read in both actions and observations
        fbs_event_reader = event_readers.FBSEventReader(observation_path, action_path, paint_cursor=paint_cursor)
        readers.append(fbs_event_reader)
    elif action_file:
        # Only read in actions
        if action_file == 'client.fbs':
            action_reader = event_readers.FBSActionReader(action_path)
        elif action_file == 'actions.jsonl':
            action_reader = event_readers.ActionReader(action_path)
        else:
            raise event_readers.InvalidEventLogfileError("Invald action_file: {}".format(reward_path))

        readers.append(action_reader)

    if reward_path:
        if reward_file == 'rewards.jsonl':
            reward_reader = event_readers.RewardReader(reward_path)
        elif reward_file == 'rewards.demo':
            reward_reader = event_readers.RewarderProtocolReader(reward_path)
        else:
            raise event_readers.InvalidEventLogfileError("Invald reward_file: {}".format(reward_path))
        readers.append(reward_reader)

    if botaction_path:
        if os.path.exists(botaction_path):
            botaction_reader = event_readers.BotactionReader(botaction_path)
            readers.append(botaction_reader)

    return event_readers.MergedEventReader(*readers)
