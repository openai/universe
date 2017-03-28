import logging
import time
import os
import json
import numpy as np
import threading
from six.moves import queue
from universe import rewarder, spaces, vectorized, pyprofile
from universe.utils import random_alphanumeric

logger = logging.getLogger(__name__)
extra_logger = logging.getLogger('universe.extra.'+__name__)


class Recording(vectorized.Wrapper):
    """
Record all action/observation/reward/info to a log file.

It will do nothing, unless given a (recording_dir='/path/to/results') argument.
recording_policy can be one of:
    'capped_cubic' will record a subset of episodes (those that are a perfect cube: 0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, and every multiple of 1000 thereafter).
    'always' records all
    'never' records none
recording_notes can be used to record hyperparameters in the log file

The format is line-separated json, with large observations stored separately in binary.

The universe-viewer project (http://github.com/openai/universe-viewer) provides a browser-based UI
for examining logs.

"""

    def __init__(self, env, recording_dir=None, recording_policy=None, recording_notes=None):
        super(Recording, self).__init__(env)
        self._log_n = None
        self._episode_ids = None
        self._step_ids = None
        self._episode_id_counter = 0
        self._env_semantics_autoreset = env.metadata.get('semantics.autoreset', False)
        self._env_semantics_async = env.metadata.get('semantics.async', False)
        self._async_write = self._env_semantics_async

        self._recording_dir = recording_dir
        if self._recording_dir is not None:
            if recording_policy == 'never' or recording_policy is False:
                self._recording_policy = lambda episode_id: False
            elif recording_policy == 'always' or recording_policy is True:
                self._recording_policy = lambda episode_id: True
            elif recording_policy == 'capped_cubic' or recording_policy is None:
                self._recording_policy = lambda episode_id: (int(round(episode_id ** (1. / 3))) ** 3 == episode_id) if episode_id < 1000 else episode_id % 1000 < 2
            else:
                self._recording_policy = recording_policy
        else:
            self._recording_policy = lambda episode_id: False
        logger.info('Running Recording wrapper with recording_dir=%s policy=%s. To change this, pass recording_dir="..." to env.configure.', self._recording_dir, recording_policy)

        self._recording_notes = {
            'env_id': env.spec.id,
            'env_metadata': env.metadata,
            'env_spec_tags': env.spec.tags,
            'env_semantics_async': self._env_semantics_async,
            'env_semantics_autoreset': self._env_semantics_autoreset,
        }
        if recording_notes is not None:
            self._recording_notes.update(recording_notes)

        if self._recording_dir is not None:
            os.makedirs(self._recording_dir, exist_ok=True)

        self._instance_id = random_alphanumeric(6)

    def _get_episode_id(self):
        ret = self._episode_id_counter
        self._episode_id_counter += 1
        return ret

    def _get_writer(self, i):
        """
        Returns a tuple of (log_fn, log_f, bin_fn, bin_f) to be written to by vectorized env channel i
        Or all Nones if recording is inactive on that channel
        """
        if self._recording_dir is None:
            return None
        if self._log_n is None:
            self._log_n = [None] * self.n
        if self._log_n[i] is None:
            self._log_n[i] = RecordingWriter(self._recording_dir, self._instance_id, i, async_write=self._async_write)
        return self._log_n[i]

    def _reset(self):
        if self._episode_ids is None:
            self._episode_ids = [None] * self.n
        if self._step_ids is None:
            self._step_ids = [None] * self.n

        for i in range(self.n):
            writer = self._get_writer(i)
            if writer is not None:
                if self._recording_notes is not None:
                    writer(type='notes', notes=self._recording_notes)
                    self._recording_notes = None
                writer(type='reset', timestamp=time.time())
            self._episode_ids[i] = self._get_episode_id()
            self._step_ids[i] = 0

        return self.env.reset()

    def _step(self, action_n):
        observation_n, reward_n, done_n, info = self.env.step(action_n)
        info_n = info["n"]

        for i in range(self.n):
            if self._recording_policy(self._episode_ids[i]):
                writer = self._get_writer(i)
                if writer is not None:
                    writer(type='step',
                           timestamp=time.time(),
                           episode_id=self._episode_ids[i],
                           step_id=self._step_ids[i],
                           action=action_n[i],
                           observation=observation_n[i],
                           reward=reward_n[i],
                           done=done_n[i],
                           info=info_n[i])
                    # Agents can later call info_n[i]['annotate'](...) to add more things to be visualized
                    info_n[i]['annotate'] = RecordingAnnotator(writer, self._episode_ids[i], self._step_ids[i])
                self._step_ids[i] += 1
            if done_n[i] and self._env_semantics_autoreset:
                self._episode_ids[i] = self._get_episode_id()
                self._step_ids[i] = 0

        return observation_n, reward_n, done_n, info

    def _close(self):
        super(Recording, self)._close()
        if self._log_n is not None:
            for i in range(self.n):
                if self._log_n[i] is not None:
                    self._log_n[i].close()
                    self._log_n[i] = None

class RecordingWriter(object):
    """
    Safe to use from multiple threads, in case your agent action generator & learning are running in parallel.
    """
    def __init__(self, recording_dir, instance_id, channel_id, async_write=True):
        self.log_fn = 'universe.recording.{}.{}.{}.jsonl'.format(os.getpid(), instance_id, channel_id)
        log_path = os.path.join(recording_dir, self.log_fn)
        self.bin_fn = 'universe.recording.{}.{}.{}.bin'.format(os.getpid(), instance_id, channel_id)
        bin_path = os.path.join(recording_dir, self.bin_fn)
        extra_logger.info('Logging to %s and %s', log_path, self.bin_fn)
        self.log_f = open(log_path, 'w')
        self.bin_f = open(bin_path, 'wb')
        # It would be better to measure memory use and block the writer when the queue is sitting on too much memory
        self.q = queue.Queue(1000)
        self.t = threading.Thread(target=self.writer_main)
        self.t.start()

    def close(self):
        self.q.put(None)

    def close_files(self):
        if self.bin_f is not None:
            self.bin_f.close()
            self.bin_f = None
        if self.log_f is not None:
            self.log_f.close()
            self.log_f = None

    def json_encode(self, obj):
        if isinstance(obj, np.ndarray):
            offset = self.bin_f.tell()
            while offset%8 != 0:
                self.bin_f.write(b'\x00')
                offset += 1
            obj.tofile(self.bin_f)
            size = self.bin_f.tell() - offset
            return {'__type': 'ndarray', 'shape': obj.shape, 'order': 'C', 'dtype': str(obj.dtype), 'npyfile': self.bin_fn, 'npyoff': offset, 'size': size}
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, RecordingAnnotator):
            return 'RecordingAnnotator'
        else:
            return obj

    def writer_main(self):
        while True:
            item = self.q.get()
            if item is None: break
            self.write_item(item)
            self.q.task_done()
        self.close_files()

    def __call__(self, **kwargs):
        pyprofile.gauge('recording.qsize', self.q.qsize())
        self.q.put(kwargs)

    def write_item(self, item):
        with pyprofile.push('recording.write'):
            l = json.dumps(item, skipkeys=True, default=self.json_encode)
            self.log_f.write(l + '\n')
            self.log_f.flush()

class RecordingAnnotator(object):
    def __init__(self, writer, episode_id, step_id):
        self.writer = writer
        self.episode_id = episode_id
        self.step_id = step_id
    def __call__(self, **kwargs):
        self.writer.__call__(type='annotate', episode_id=self.episode_id, step_id=self.step_id, **kwargs)
