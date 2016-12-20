import json
import logging
import os
import re
import requests
import six
import six.moves.urllib.parse as urlparse
import threading
import time

import gym
from gym import scoreboard
from gym.utils import reraise

from universe import error, utils
from universe.remotes import remote

if six.PY2:
    import Queue as queue
else:
    import queue

logger = logging.getLogger(__name__)
extra_logger = logging.getLogger('universe.extra.'+__name__)

# Using gym for this
_api_key = 'tyytjgq3envte2j9yv2e-{}'.format(os.environ.get('OPENAI_USER', os.environ.get('USER')))
allocator_base = 'http://allocator.sci.openai-tech.com'
# gym_base_url = 'http://api.gym.sci.openai-tech.com'

class Stop(Exception):
    pass

class RequestError(Exception):
    def __init__(self, message, status_code=None):
        super(RequestError, self).__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code

class FatalError(RequestError):
    pass

class AllocatorManager(threading.Thread):
    daemon = True

    def __init__(self, client_id, base_url=allocator_base,
                 address_type=None, start_timeout=None, api_key=None,
                 runtime_id=None, tag=None, params=None, placement=None,
                 use_recorder_ports=False,
    ):
        super(AllocatorManager, self).__init__()
        self.label = 'AllocatorManager'

        self.supports_reconnect = True
        self.connect_vnc = True
        self.connect_rewarder = True

        if address_type is None: address_type = 'public'
        if address_type not in ['public', 'pod', 'private']:
            raise error.Error('Bad address type specified: {}. Must be public, pod, or private.'.format(address_type))

        self.tag = tag
        self.client_id = client_id
        self.address_type = address_type

        if start_timeout is None:
            start_timeout = 20 * 60
        self.start_timeout = start_timeout
        self.params = params
        self.placement = placement
        self.use_recorder_ports = use_recorder_ports

#         if base_url is None:
#             base_url = scoreboard.api_base
#         if base_url is None:
#             base_url = gym_base_url
#         if api_key is None:
#             api_key = scoreboard.api_key
#         if api_key is None:
#             raise gym.error.AuthenticationError("""You must provide an OpenAI Gym API key.

# (HINT: Set your API key using "gym.scoreboard.api_key = .." or "export OPENAI_GYM_API_KEY=..."). You can find your API key in the OpenAI Gym web interface: https://gym.openai.com/settings/profile.""")

        if api_key is None:
            api_key = _api_key
        self._requestor = AllocatorClient(self.label, api_key, base_url=base_url)
        self.base_url = base_url

        # These could be overridden on a per-allocation basis, if you
        # want heterogeoneous envs. We don't support those currently
        # in the higher layers, but this layer could support it
        # easily.
        self.runtime_id = runtime_id
        self.tag = tag

        self.pending = {}

        self.error_buffer = utils.ErrorBuffer()
        self.requests = queue.Queue()
        self.ready = queue.Queue()

        self._reconnect_history = {}
        self._sleep = 1

    @classmethod
    def from_remotes(cls, client_id, remotes, runtime_id, start_timeout, tag, api_key, use_recorder_ports):
        parsed = urlparse.urlparse(remotes)
        if not (parsed.scheme == 'http' or parsed.scheme == 'https'):
            raise error.Error('AllocatorManager must start with http:// or https://: {}'.format(remotes))

        base_url = parsed.scheme + '://' + parsed.netloc
        if parsed.path:
            base_url += '/' + parsed.path
        query = urlparse.parse_qs(parsed.query)

        n = query.get('n', [1])[0]
        cpu = query.get('cpu', [None])[0]
        if cpu is not None:
            cpu = float(cpu)
        placement = query.get('address', ['public'])[0]

        params = {}
        if tag is not None: params['tag'] = tag
        if cpu is not None: params['cpu'] = cpu

        return cls(client_id=client_id, runtime_id=runtime_id, base_url=base_url, start_timeout=start_timeout, params=params, placement=placement, api_key=api_key, use_recorder_ports=use_recorder_ports), int(n)

    def pop(self, n=None):
        """Call from main thread. Returns the list of newly-available (handle, env) pairs."""
        self.error_buffer.check()

        envs = []

        if n is None:
            while True:
                try:
                    envs += self.ready.get(block=False)
                except queue.Empty:
                    break
        else:
            sync_timeout = 10 * 60
            start = time.time()

            wait_time = 1
            while len(envs) < n:
                try:
                    extra_logger.info('[%s] Waiting for %d envs, currently at %d, sleeping for %d', self.label, n, len(envs), wait_time)
                    envs += self.ready.get(timeout=wait_time)
                except queue.Empty:
                    self.error_buffer.check()
                wait_time = min(wait_time * 2, 30)
                delta = time.time() - start
                if delta > sync_timeout:
                    raise FatalError("Waited %.0fs to obtain envs, timeout was %.0fs. (Obtained %d/%d envs.)" % (delta, sync_timeout, len(envs), n))

        return envs

    def allocate(self, handles, initial=False, params={}):
        """Call from main thread. Initiate a request for more environments"""
        assert all(re.search('^\d+$', h) for h in handles), "All handles must be numbers: {}".format(handles)
        self.requests.put(('allocate', (handles, initial, params)))

    def close(self):
        self.requests.put(('close', ()))

    def run(self):
        try:
            self._run()
        except Stop:
            pass
        except Exception as e:
            self.error_buffer.record(e)

    def _run(self):
        while True:
            self._process_requests()
            self._poll()

    def _process_requests(self):
        while True:
            try:
                method, args = self.requests.get(timeout=self._sleep)
            except queue.Empty:
                break
            else:
                if method == 'allocate':
                    handles, initial, params = args
                    self._allocate(handles, initial, params)
                elif method == 'close':
                    raise Stop

    def _allocate(self, handles, initial, params):
        self._sleep = 1

        _params = self.params.copy()
        _params.update(params)

        for handle in handles:
            history = self._reconnect_history.get(handle, [])
            history.append(time.time())
            floor = time.time() - 5 * 60
            history = [entry for entry in history if entry > floor]
            if len(history) > 5:
                raise error.Error('Tried reallocating a fresh remote at index {} a total of {} times in the past 5 minutes (at {}). Please examine the logs to determine why the remotes keep failing.'.format(handle, len(history), history))
            self._reconnect_history[handle] = history

        assert all(re.search('^\d+$', h) for h in handles), "All handles must be numbers: {}".format(handles)
        allocation = self.with_retries(self._requestor.allocation_create,
            client_id=self.client_id,
            runtime_id=self.runtime_id,
            placement=self.placement,
            params=_params,
            handles=handles,
            initial=initial,
        )
        news = len([entry for entry in allocation['info']['n'] if entry['new']])
        extra_logger.info('[%s] Received allocation with %s new and %s existing envs: %s', self.label, news, len(allocation['info']['n']) - news, allocation)

        assert len(allocation['env_n']) <= len(handles), "Received more envs than requested: allocation={} handles={}".format(allocation, handles)
        _, pending = self._handle_allocation(allocation)

        for env in pending:
            self.pending[env['name']] = {
                'handle': env['handle'],
                'params': params,
                'received_at': time.time()
            }

    def _poll(self):
        self._sleep = min(20, self._sleep + 2)

        if len(self.pending) == 0:
            return

        for name, spec in self.pending.items():
            delta = time.time() - spec['received_at']
            if delta > self.start_timeout:
                raise error.TimeoutError('Waited {}s for {} to get an IP, which exceeds start_timeout of {}'.format(delta, name, self.start_timeout))

        names = list(self.pending.keys())
        # This really should be an allocation_get, but it's possible
        # the pods list will be long. So it's either GET with a body,
        # or POST what should really be a GET. We do the latter.
        allocation = self.with_retries(self._requestor.allocation_refresh, self.client_id, names=names)
        assert len(allocation['env_n']) <= len(names), "Received more envs than requested: allocation={} names={}".format(allocation, names)

        # Handle any envs which have gone missing
        result = set(env['name'] for env in allocation['env_n'])
        dropped = [p for p in self.pending.keys() if p not in result]
        if len(dropped) > 0:
            logger.info('Pending remote envs %s were not returned by the allocator (only %s were returned). Assuming the missing ones have gone down and requesting replacements.', dropped, list(result))
            for d in dropped:
                spec = self.pending.pop(d)
                self._allocate(dropped, False, spec['params'])

        # Handle successful allocations
        self._handle_allocation(allocation, pop=True)

    def _handle_allocation(self, allocation, pop=False):
        ready = []
        not_ready = []
        for alloc_env in allocation['env_n']:
            if alloc_env['status'] != 'allocated':
                not_ready.append(alloc_env)
                continue
            if pop:
                self.pending.pop(alloc_env['name'])
            vnc_address = alloc_env['vnc_recorder_address'] if self.use_recorder_ports else alloc_env['vnc_address']
            rewarder_address = alloc_env['rewarder_recorder_address'] if self.use_recorder_ports else alloc_env['rewarder_address']
            env = remote.Remote(
                name=alloc_env['name'],
                handle=alloc_env['handle'],
                vnc_address=vnc_address,
                vnc_password=alloc_env['vnc_password'],
                rewarder_address=rewarder_address,
                rewarder_password=alloc_env['rewarder_password'],
            )
            ready.append(env)

        if len(ready) > 0:
            extra_logger.info('[%s] The following envs now have IPs, but still may take time to boot: %s', self.label, ready)
            self.ready.put(ready)

        return ready, not_ready

    def with_retries(self, method, *args, **kwargs):
        timeout = 20 * 60
        start = time.time()

        i = 0
        while True:
            try:
                return method(*args, **kwargs)
            except FatalError as e:
                logger.error('[%s] %s', self.label, e)
                self.error_buffer.record(e)
                raise
            except Exception as e:
                delta = time.time() - start
                if delta > timeout:
                    raise error.TimeoutError('Have been unable to connect to the allocator at {} for {}s. Giving up. Last error: {}'.format(self.base_url, delta, e))
                i += 1

                sleep = min(2**i, 60)
                time.sleep(sleep)
                logger.error('[%s] Error making request to allocator: %s. Will retry in %ss (and timeout in %.0fs)', self.label, e, sleep, start + timeout - time.time())

class AllocatorClient(object):
    def __init__(self, label, api_key, base_url):
        self.label = label
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-type': 'application/json'})
        self.request_timeout = 80

    def _handle_resp(self, resp):
        try:
            parsed = resp.json()
        except ValueError as e:
            if resp.status_code == 500:
                raise RequestError(message="500 Internal Error (retrying automatically): response={}".format(resp.content), status_code=resp.status_code)
            elif resp.status_code == 503:
                raise RequestError(message="503 Error from server (allocator is probably overloaded): response={}".format(resp.content), status_code=resp.status_code)
            elif resp.status_code == 200:
                raise RequestError(message="Response from server: status_code={} response={}".format(resp.status_code, resp.content), status_code=resp.status_code)
            else:
                raise RequestError(message="Error from server: status_code={} response={}".format(resp.status_code, resp.content), status_code=resp.status_code)

        if resp.status_code == 200:
            return parsed
        elif 'detail' in parsed:
            raise FatalError(message=parsed['detail'], status_code=resp.status_code)
        else:
            raise RequestError(message='Malformed response from allocator, missing "detail" key: {}'.format(parsed), status_code=resp.status_code)

    def _post_request(self, route, data, description):
        url = urlparse.urljoin(self.base_url, route)
        extra_logger.info('[%s] %s: POST %s: %s', self.label, description, url, json.dumps(data))
        resp = self.session.post(urlparse.urljoin(self.base_url, route),
                                 data=json.dumps(data), auth=(self.api_key, ''),
                                 timeout=self.request_timeout,
        )
        return self._handle_resp(resp)

    def _delete_request(self, route):
        url = urlparse.urljoin(self.base_url, route)
        extra_logger.info("[%s] DELETE %s", self.label, url)
        resp = self.session.delete(url, auth=(self.api_key, ''), timeout=self.request_timeout)
        return self._handle_resp(resp)

    def _get_request(self, route):
        url = urlparse.urljoin(self.base_url, route)
        extra_logger.info("[%s] GET %s", self.label, url)
        resp = self.session.get(url, auth=(self.api_key, ''), timeout=self.request_timeout)
        return self._handle_resp(resp)

    def allocation_create(self, client_id, runtime_id, handles, params={}, placement='public', initial=False):
        route = '/v1/allocations'
        data = {'client': client_id, 'runtime': runtime_id, 'params': params, 'handles': handles, 'initial': initial, 'placement': placement}
        logger.info('Requesting %s environment%s from %s: %s', len(handles), 's' if len(handles) != 1 else '', self.base_url, data)
        resp = self._post_request(route, data, description='requesting new allocation')
        return resp

    def allocation_refresh(self, id, names):
        route = '/v1/allocations/{}'.format(id)
        resp = self._post_request(route, data={'names': names}, description='refreshing existing allocation')
        return resp

    def allocation_delete(self, id):
        route = '/v1/allocations/{}'.format(id)
        resp = self._post_request(route, {})
        return resp
