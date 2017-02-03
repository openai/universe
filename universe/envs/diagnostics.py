import collections
import fastzbarlight
import itertools
import logging
from multiprocessing import pool
import numpy as np
import time
import threading
# import psutil
import sys
from collections import namedtuple
from gym.utils import reraise

import re
from universe import error, pyprofile, spaces

# TODO: prefix the loggers

logger = logging.getLogger(__name__)
extra_logger = logging.getLogger('universe.extra.'+__name__)

def show(ob):
    from PIL import Image
    Image.fromarray(ob).show()

def standard_error(ary, axis, scale=1):
    ary = np.array(ary) * scale
    if len(ary) > 1:
        return np.std(ary, axis=axis) / np.sqrt(len(ary) - 1)
    else:
        return np.std(ary, axis=axis)

def extract_timestamp(observation):
    total = 0
    for byte in observation[0]:
        total = 256 * total + byte
    for byte in observation[1]:
        total = 256 * total + byte

    timestamp = total/1000.
    return timestamp

class MetadataDecoder(object):
    @classmethod
    def build(cls, metadata_encoding, pool, qr_pool, label):
        metadata_encoding = metadata_encoding.copy()
        type = metadata_encoding.pop('type')
        if type == 'qrcode':
            return QRCodeMetadataDecoder(label=label, pool=pool, qr_pool=qr_pool, **metadata_encoding)
        elif type == 'pixels':
            return PixelsMetadataDecoder(label=label)
        else:
            raise error.Error('Invalid encoding: {}'.format(type))

class AsyncDecode(object):
    pool = None

    def __init__(self, pool, qr_pool, method, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self._last_img = None

        self.method = method
        self.results = []
        self.deque = collections.deque()
        self.pool = pool
        self.qr_pool = qr_pool

    def __call__(self, img, available_at):
        # Choose the return value
        if len(self.deque) > 0 and self.deque[0].ready():
            last = self.deque.popleft()
            res = last.get()
            if res is not None:
                pyprofile.timing('vnc_env.diagnostics.async_decode.latency', time.time() - res['available_at'])
        else:
            res = False

        pyprofile.gauge('vnc_env.diagnostics.async_decode.queue_depth', len(self.deque))

        # Just grayscale it by keeping only one component. Should be
        # good enough as this region is black and white anyway.
        grayscale = img[self.y:self.y+self.height, self.x:self.x+self.width, 0]

        # Apply processing if needed
        match = np.array_equal(self._last_img, grayscale)
        if not match:
            pyprofile.incr('vnc_env.diagnostics.async_decode.schedule')
            # sneakily copy if numpy hasn't, so it can be cached
            self._last_img = np.ascontiguousarray(grayscale)
            async = self.qr_pool.apply_async(self.method, (self._last_img, time.time(), available_at))
            self.deque.append(async)
        else:
            pyprofile.incr('vnc_env.diagnostics.async_decode.cache_hit')

        return res

class QRCodeMetadataDecoder(MetadataDecoder):
    def __init__(self, pool, qr_pool, x, y, width, height, label):
        self.flag_synchronous = False

        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label

        self.decode = AsyncDecode(pool, qr_pool, self._decode, x, y, width, height)

    def _decode(self, observation, start, available_at):
        # This method gets wrapped by AsyncDecode.__call__
        with pyprofile.push('vnc_env.diagnostics.QRCodeMetadataDecoder.qr_code_scanner'):

            encoded = fastzbarlight.qr_code_scanner(observation.tobytes(), self.width, self.height)
        if encoded is None:
            # Failed to parse!
            return
        if encoded.startswith(b'v1:'):
            encoded = encoded.decode('utf-8')
            if len(encoded) != len('v1:') + 12 + 12:
                raise error.Error('Bad length for metadata from enviroment: {}'.format(encoded))

            encoded = encoded[len('v1:'):]
            last_update = int(encoded[:12], 16) / 1000.0
            last_action = int(encoded[12:24], 16) / 1000.

            return {
                # Timestamp on the image
                'now': last_update,
                # When the last probe was received
                'probe_received_at': last_action,
                'processing_start': start,
                'processing_end': time.time(),
                'available_at': available_at,
            }
        else:
            raise error.Error('Bad version string for metadata from environment: {}'.format(encoded))


class PixelsMetadataDecoder(MetadataDecoder):
    def __init__(self, label):
        self.flag_synchronous = True

        self.anchor = np.array([
            [(0x12, 0x34, 0x56), (0x78, 0x90, 0xab)],
            [(0x23, 0x45, 0x67), (0x89, 0x0a, 0xbc)],
        ], dtype=np.uint8)

        self.location = None
        self.last_search_metadata = 0
        self.label = label

    def _check_location(self, observation, location):
        y, x = location
        return np.all(observation[y:y+2, x:x+2] == self.anchor)

    def _find_metadata_location(self, observation):
        ys, xs = np.where(np.all(observation == self.anchor[0, 0], axis=-1))
        if len(ys) == 0:
            extra_logger.info('[%s] Could not find metadata anchor pixel', self.label)
            return False

        # TODO: handle multiple hits
        assert len(ys) == 1
        location = (ys[0], xs[0])

        assert self._check_location(observation, location)
        extra_logger.info('[%s] Found metadata anchor pixel: %s', self.label, location)
        return location

    def _should_search_metadata(self):
       return time.time() - self.last_search_metadata > 1

    def decode(self, observation, available_at=None):
        start = time.time()

        # metadata pixel location hasn't been initialized or it has moved
        if not self.location or not self._check_location(observation,
                                                         self.location):
            # only search for metadata occasionally
            if self._should_search_metadata():
                self.location = self._find_metadata_location(observation)
                self.last_search_metadata = time.time()

        if not self.location:
            return False  # False translates to None in DiagnosticsInstance

        y, x = self.location
        now = extract_timestamp(observation[y, x+2:x+4])
        probe_received_at = extract_timestamp(observation[y, x+4:x+6])

        return {
            'now': now,
            'probe_received_at': probe_received_at,
            'processing_start': start,
            'processing_end': time.time(),
            'available_at': available_at,
        }

class Diagnostics(object):
    def __init__(self, n, probe_key, ignore_clock_skew=False, metadata_encoding=None, disable_action_probes=False):
        # Each QR code takes about 1ms (and updates at 5fps). We do
        # our best to ensure the QR is processed in time for the next
        # step call (n/16 would put us right at the threshold).
        self.pool = pool.ThreadPool(max(int(n/4), 1))
        self.qr_pool = pool.ThreadPool(max(int(n/8), 1))
        self.lock = threading.RLock()

        self.instance_n = [None] * n
        self.ignore_clock_skew = ignore_clock_skew
        self.disable_action_probes = disable_action_probes

        self.metadata_encoding = metadata_encoding

        self.update(probe_key=probe_key, metadata_encoding=metadata_encoding)

    # only used in flashgames right now
    def update(self, probe_key, metadata_encoding):
        self.probe_key = probe_key
        self.metadata_encoding = metadata_encoding

        for instance in self.instance_n:
            if instance is not None:
                instance.update(probe_key=self.probe_key, metadata_encoding=self.metadata_encoding)

    def connect(self, i, network=None, label=None):
        # This should technically be synchronized
        self.instance_n[i] = DiagnosticsInstance(i, network, self.probe_key, self.ignore_clock_skew, self.metadata_encoding, disable_action_probes=self.disable_action_probes, qr_pool=self.qr_pool, pool=self.pool, label=label)

    def close(self, i=None):
        if i is not None:
            self.instance_n[i] = None
        else:
            self.pool.close()
            self.qr_pool.close()
            for i in range(len(self.instance_n)):
                self.close(i)
            self.instance_n = None

    def add_probe(self, action_n, mask_n):
        if self.disable_action_probes or self.instance_n is None:
            return

        for instance, action, mask in zip(self.instance_n, action_n, mask_n):
            # Important that masking prevents us from adding probes. (This
            # avoids us e.g. filling in backticks into text boxes as the
            # environment boots.)
            if mask and instance:
                instance.add_probe(action)

    def add_metadata(self, observation_n, info_n, available_at=None):
        """Mutates the info_n dictionary."""
        if self.instance_n is None:
            return

        with pyprofile.push('vnc_env.diagnostics.Diagnostics.add_metadata'):
            async = self.pool.imap_unordered(
                self._add_metadata_i,
                zip(self.instance_n, observation_n, info_n, [available_at] * len(observation_n)))
            list(async)

    def _add_metadata_i(self, args):
        instance, observation, info, now = args
        if instance is None or observation is None:
            return
        instance.add_metadata(observation, info, now)

    def extract_metadata(self, observation_n):
        return [instance._extract_metadata(observation)
                for instance, observation in zip(self.instance_n, observation_n)]

    def clear_probes_when_done(self, done_n):
        if self.instance_n is None: # if we've been closed there's nothing to do
            return
        for instance, done in zip(self.instance_n, done_n):
            if done:
                instance.clear_probe()

class DiagnosticsInstance(object):
    anchor = np.array([
        [(0x12, 0x12, 0x12), (0x78, 0x78, 0x78)],
        [(0x23, 0x23, 0x23), (0x89, 0x89, 0x89)],
    ], dtype=np.uint8)
    zero_clock_skew = np.zeros([2])

    def __init__(self, i, network, probe_key, ignore_clock_skew, metadata_encoding, disable_action_probes, pool, qr_pool, label=None):
        '''
        network - either Network() object used to get clock skew, or None.

                  If None, we skip measuring clock skew, and skip measuring
                  diagnostics which rely on clock skew.
        '''
        if network is None:
            assert ignore_clock_skew
        self.ignore_clock_skew = ignore_clock_skew

        self.label = label
        self.i = i
        self.network = network

        self.probe_sent_at = None # local time
        self.probe_received_at = None # remote time
        self.action_latency_skewed = None
        self.last_observation_timestamp = None
        self.disable_action_probes = disable_action_probes

        self.pool = pool
        self.qr_pool = qr_pool
        self.could_read_metadata = None
        self.update(probe_key=probe_key, metadata_encoding=metadata_encoding)

    def update(self, probe_key, metadata_encoding):
        self.probe = [
                spaces.KeyEvent(probe_key, down=True).compile(),
                spaces.KeyEvent(probe_key, down=False).compile(),
        ]

        if metadata_encoding is not None:
            self.metadata_decoder = MetadataDecoder.build(metadata_encoding, pool=self.pool, qr_pool=self.qr_pool, label=self.label)
        else:
            self.metadata_decoder = None

    def clear_probe(self):
        self.probe_sent_at = None
        self.probe_received_at = None

    def add_probe(self, action):
        if self.network is not None and not self.network.active():
            return

        if self.probe_sent_at is not None and self.probe_sent_at + 10 < time.time():
            extra_logger.warn('[%s] Probe to determine action latency timed out (was sent %s). (This is harmless, but worth knowing about.)', self.label, self.probe_sent_at)
            self.probe_sent_at = None
        if self.probe_sent_at is None:
            extra_logger.debug('[%s] Sending out new action probe: %s', self.label, self.probe)
            self.probe_sent_at = time.time()
            action += self.probe
        assert self.probe_sent_at is not None

    def add_metadata(self, observation, info, available_at=None):
        """Extract metadata from a pixel observation and add it to the info
        """
        observation = observation['vision']
        if observation is None: return
        if self.network is not None and not self.network.active():
            return
        elif self.metadata_decoder is None:
            return
        elif observation is None:
            return
        # should return a dict with now/probe_received_at keys
        with pyprofile.push('vnc_env.diagnostics.DiagnosticsInstance.add_metadata.decode'):
            metadata = self.metadata_decoder.decode(observation, available_at=available_at)

        if metadata is False:
            # No metadata ready, though it doesn't mean parsing failed
            metadata = None
        elif metadata is None:
            if self.could_read_metadata:
                self.could_read_metadata = False
                extra_logger.info('[%s] Stopped being able to read metadata (expected when environment resets)', self.label)
        elif not self.could_read_metadata:
            self.could_read_metadata = True
            extra_logger.info('[%s] Started being able to read metadata', self.label)

        if self.metadata_decoder.flag_synchronous and metadata is not None:
            info['diagnostics.image_remote_time'] = metadata['now']

        local_now = time.time()

        if self.network is None:
            # Assume the clock skew is zero. Should only be run on the
            # same machine as the VNC server, such as the universe
            # instance inside of the environmenth containers.
            real_clock_skew = self.zero_clock_skew
        else:
            # Note: this is a 2-length vector of (min, max), so anything added to
            # it is also going to be a 2-length vector.
            # Most of the diagnostics below are, but you have to look carefully.
            real_clock_skew = self.network.reversed_clock_skew()

        # Store real clock skew here
        info['stats.gauges.diagnostics.clock_skew'] = real_clock_skew
        if self.ignore_clock_skew:
            clock_skew = self.zero_clock_skew
        else:
            clock_skew = real_clock_skew

        if metadata is not None:
            # We'll generally update the observation timestamp infrequently
            if self.last_observation_timestamp == metadata['now']:
                delta = None
            else:
                # We just got a new timestamp in the observation!
                self.last_observation_timestamp = metadata['now']
                observation_now = metadata['now']
                delta = observation_now - metadata['available_at']

                # Subtract *local* time it was received from the *remote* time
                # displayed. Negate and reverse order to fix time ordering.
                info['stats.gauges.diagnostics.lag.observation'] = -(delta + clock_skew)[[1, 0]]

            # if self.network is None:
            #     # The rest of diagnostics need the network, so we're done here
            #     return

            probe_received_at = metadata['probe_received_at']
            if probe_received_at == 0 or self.disable_action_probes:
                # Happens when the env first starts
                self.probe_received_at = None
            elif self.probe_received_at is None: # this also would work for the equality case
                self.probe_received_at = probe_received_at
            elif self.probe_received_at != probe_received_at and self.probe_sent_at is None:
                logger.info('[%s] Probe is marked as received at %s, but probe_sent_at is None. This is surprising. (HINT: do you have multiple universe instances talking to the same environment?)', self.label, probe_received_at)
            elif self.probe_received_at != probe_received_at:
                extra_logger.debug('[%s] Next probe received: old=%s new=%s', self.label, self.probe_received_at, probe_received_at)
                self.probe_received_at = probe_received_at
                # Subtract the *local* time we sent it from the *remote* time it was received
                self.action_latency_skewed = probe_received_at - self.probe_sent_at
                self.probe_sent_at = None

            if self.action_latency_skewed:
                action_lag = self.action_latency_skewed + clock_skew
                self.action_latency_skewed = None
            else:
                action_lag = None
            info['stats.gauges.diagnostics.lag.action'] = action_lag

        local_now = time.time()
        # Look at when the remote believed it parsed the score (not
        # all envs send this currently).
        #
        # Also, if we received no new rewards, then this values is
        # None. This could indicate a high reward latency (bad,
        # uncommon), or that the agent is calling step faster than new
        # rewards are coming in (good, common).
        remote_score_now = info.get('rewarder.lag.observation.timestamp')
        if remote_score_now is not None:
            delta = remote_score_now - local_now
            info['stats.gauges.diagnostics.lag.reward'] = -(delta + clock_skew)[[1, 0]]

        # Look at when the remote send the message, so we know how
        # long it's taking for messages to get to us.
        rewarder_message_now = info.get('reward_buffer.remote_time')
        if rewarder_message_now:
            delta = rewarder_message_now - local_now
            info['stats.gauges.diagnostics.lag.rewarder_message'] = -(delta + clock_skew)[[1, 0]]


def extract_n_m(dict_n_m, key):
    output = []
    for dict_n in dict_n_m:
        layer = []
        for dict in dict_n:
            layer.append(dict[key])
        output.append(layer)
    return np.array(output)


# class ChromeProcessInfo(object):
#     proc_regex = re.compile('.*(chrome|Chrome|nacl_helper).*')

#     def add_system_stats(self, info, now):
#         """TODO: This needs be moved to universe-envs and run there. Otherwise it only works if the env and agent
#             are on the same machine. In addition a new rpc call, rpc.env.diagnostics, should be added to return
#             data to the agent periodically.
#         """
#         start = time.time()

#         # CPU
#         cpu_percent = psutil.cpu_percent()
#         info['diagnostics.env.cpu.percent'] = cpu_percent
#         cpu_cores_percent = psutil.cpu_percent(percpu=True)
#         num_cores = len(cpu_cores_percent)
#         info['diagnostics.env.cpu.percent.all_cores'] = cpu_percent / num_cores
#         info['diagnostics.env.cpu.percent.each_core'] = cpu_cores_percent
#         info['diagnostics.env.cpu.num_cores'] = num_cores

#         # MEMORY
#         mem = psutil.virtual_memory()
#         info['diagnostics.env.memory.percent'] = mem.percent
#         info['diagnostics.env.memory.total'] = mem.total
#         info['diagnostics.env.memory.available'] = mem.available

#         # NETWORK
#         if self.last_measured_at is not None:
#             elapsed_ms = (now - self.last_measured_at) * 1000.
#             current = psutil.net_io_counters()
#             dl = (current.bytes_recv - self.system_network_counters.bytes_recv) / elapsed_ms
#             ul = (current.bytes_sent - self.system_network_counters.bytes_sent) / elapsed_ms
#             info['diagnostics.env.network.download_bytes_ps'] = dl * 1000.
#             info['diagnostics.env.network.upload_bytes_ps'] = ul * 1000.
#             self.system_network_counters = current

#         # CHROME
#         if self.chrome_last_measured_at is None or (time.time() - self.chrome_last_measured_at) > 30:
#             # Fetch every 30 seconds
#             self.chrome_last_measured_at = time.time()
#             logger.info("Measuring Chrome process statistics")
#             chrome_info = ChromeProcessInfo()
#             chrome_info = best_effort(chrome_info.fetch, num_cores)
#             if chrome_info is not None:
#                 self.chrome_info = chrome_info

#         if self.chrome_info is not None:
#             self._populate_chrome_info(self.chrome_info, info)

#         # TODO: Add GPU stats

#         pyprofile.push('diagnostics.system_stats')

#     def _populate_chrome_info(self, chrome_info, info):
#         pyprofile.push('diagnostics.chrome_process_info.process_iter')
#         pyprofile.push('diagnostics.chrome_process_info.total')
#         info['diagnostics.chrome.age'] = chrome_info.age
#         info['diagnostics.chrome.cpu.time'] = chrome_info.cpu_time
#         info['diagnostics.chrome.cpu.percent'] = chrome_info.cpu_percent
#         info['diagnostics.chrome.cpu.percent.all_cores'] = chrome_info.cpu_percent_all_cores
#         info['diagnostics.chrome.cpu.percent.all_cores_all_time'] = chrome_info.cpu_percent_all_cores_all_time
#         info['diagnostics.chrome.num_processes'] = len(chrome_info.processes)

#     def __init__(self):
#         self.cpu_time = 0.
#         self.cpu_percent = 0.
#         self.min_create_time = None
#         self.visited_pids = set()
#         self.processes = []
#         self.time_to_get_procs = None
#         self.total_time_to_measure = None
#         self.age = None
#         self.cpu_percent_all_cores_all_time = None
#         self.cpu_percent_all_cores = None

#     def fetch(self, num_cores):
#         start = time.time()
#         start_process_iter = time.time()
#         procs = list(psutil.process_iter())
#         self.time_to_get_procs = time.time() - start_process_iter
#         for proc in procs:
#             try:
#                 name = proc.name()
#                 if self.proc_regex.match(name):
#                     self._fetch_single(proc, name)
#                     # N.B. Don't read children. defunct processes make this take 4ever.
#                     # Child processes are all uncovered by initial scan.
#             except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
#                 pass
#         self.total_time_to_measure = time.time() - start
#         if self.min_create_time is None:
#             self.age = 0
#         else:
#             self.age = time.time() - self.min_create_time
#         self.cpu_percent_all_cores_all_time = 100. * self.cpu_time / (self.age * num_cores)
#         self.cpu_percent_all_cores = self.cpu_percent / num_cores
#         return self

#     def _fetch_single(self, proc, name):
#         if proc.pid in self.visited_pids:
#             return
#         try:
#             cpu_times = proc.cpu_times()
#             cpu_percent = proc.cpu_percent()
#             created = proc.create_time()
#             if self.min_create_time is None:
#                 self.min_create_time = created
#             else:
#                 self.min_create_time = min(created, self.min_create_time)

#             cpu_time = cpu_times.user + cpu_times.system
#             proc_info = namedtuple('proc_info', 'name cpu_time cpu_percent created age')
#             proc_info.name = name
#             proc_info.cpu_time = cpu_time
#             proc_info.cpu_percent = cpu_percent
#             proc_info.created = created
#             proc_info.age = time.time() - created
#             proc_info.pid = proc.pid
#             self.processes.append(proc_info)

#             # Totals
#             self.cpu_time += cpu_time
#             self.cpu_percent += cpu_percent
#             self.visited_pids.add(proc.pid)


#         except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
#             pass
