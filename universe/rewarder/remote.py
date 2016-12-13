# loaded inside of the environments

import logging
import os
from universe import pyprofile
import sys
is_py2 = sys.version[0] == '2'
if is_py2:
    import Queue as queue
else:
    import queue as queue
import threading
import time
import ujson
import collections

from autobahn.twisted import websocket
from universe.twisty import reactor

from universe import error, utils

logger = logging.getLogger(__name__)

class Exit(Exception):
    pass

class RewarderProtocol(websocket.WebSocketServerProtocol):
    connections = None

    def onConnect(self, request):
        if not os.path.exists('/usr/local/openai/privileged_state/password'):
            raise error.Error('No such file: /usr/local/openai/privileged_state/password. (HINT: did the init script run /app/universe-envs/base/openai-setpassword?)')
        with open('/usr/local/openai/privileged_state/password') as f:
            password = f.read().strip()

        self._message_id = 0
        self._request = request
        self._observer = request.headers.get('openai-observer') == 'true'
        self.password = password

        logger.info('Client connecting: peer=%s observer=%s', request.peer, self._observer)

    def authenticate(self, request):
        # Ugly, but it'll have to do for now.
        authorization = request.headers.get('authorization')
        if authorization is None:
            logger.info('REJECT REASON: No authorization header supplied: %s', request.headers)
            self.reject('No authorization header supplied. You must supply a basic authentication header.')
            return
        basic = utils.basic_auth_decode(authorization)
        if basic is None:
            logger.info('REJECT REASON: Invalid basic auth header: %s', request.headers)
            self.reject('Could not parse authorization header. You must supply a basic authentication header.')
            return
        username, password = basic
        if username != self.password:
            logger.info('REJECT REASON: Invalid password: %r (%s expected; %s)', username, self.password, request.headers)
            self.reject('Invalid password: {!r}. If you are using the allocator, you should see your password in the logs; if spinning up an environment by hand, it defaults to "openai". Connect as vnc://<ip>:<port>?password=<password>.'.format(username))
            return

    def onOpen(self):
        logger.info('WebSocket connection established')
        # Need to wait until onOpen to send messages
        self.authenticate(self._request)
        self.factory.agent_conn._register(self, observer=self._observer)

        # Inform the agent about the current env status
        env_info = self.factory.agent_conn.env_status.env_info()
        # Immediately upon connection, let the agent know the current
        # status.
        self.factory.agent_conn.send_env_describe_from_env_info(
            env_info,
        )

    def onMessage(self, payload, isBinary):
        if not self.factory.agent_conn.check_message(self):
            return

        assert not isBinary, "Binary websocket not supported"
        payload = ujson.loads(payload)

        context = {
            'start': time.time(),
            'conn': self,
        }
        latency = context['start'] - payload['headers']['sent_at']
        pyprofile.incr('rewarder_protocol.messages')
        pyprofile.incr('rewarder_protocol.messages.{}'.format(payload['method']))

        pyprofile.timing('rewarder_protocol.latency.rtt.skew_unadjusted', 2*latency)
        if latency < 0:
            pyprofile.incr('rewarder_protocol.latency.rtt.skew_unadjusted.negative')

        if payload['method'] == 'v0.env.reset':
            logger.info('Received reset message: %s', payload)
            self.factory.agent_conn.control_buffer.recv_rpc(context, payload)
        elif payload['method'] == 'v0.control.ping':
            logger.debug('Received ping message: %s', payload)
            parent_message_id = payload['headers']['message_id']
            headers = {'parent_message_id': parent_message_id}
            self.send_message('v0.reply.control.ping', {}, headers)
        else:
            logger.warn('Received unsupported message: %s', payload)

    def onClose(self, wasClean, code, reason):
        logger.info('WebSocket connection closed: %s', reason)
        self.factory.agent_conn._unregister(self)

    def send_message(self, method, body, headers):
        id = self._message_id

        self._message_id += 1
        new_headers = {
            'message_id': id,
            'sent_at': time.time(),
        }
        if headers:
            new_headers.update(headers)

        payload = {
            'method': method,
            'body': body,
            'headers': new_headers,
        }

        # This is a bit ugly, but decide how much we care
        if (method != 'v0.reply.control.ping' and 'parent_message_id' in new_headers) or\
           method == 'v0.connection.close':
            logger.info('Sending rewarder message: %s', payload)
        else:
            logger.debug('Sending rewarder message: %s', payload)

        self.sendMessage(ujson.dumps(payload).encode('utf-8'), False)

    def reject(self, message):
        self.send_message('v0.connection.close', {'message': message}, {})
        self.sendClose(code=1000, reason=message)
        self.transport.loseConnection()

class ControlBuffer(object):
    def __init__(self, cv):
        self.buf = queue.Queue()
        self.cv = cv

    def recv_rpc(self, context, payload):
        """Call from any thread"""
        logger.debug("Adding RPC payload to ControlBuffer queue: %s", payload)
        self.buf.put(('rpc', (context, payload)))
        with self.cv:
            self.cv.notifyAll()

    def client_disconnect(self, conn, stats):
        self.buf.put(('client_disconnect', (conn, stats)))

    def get(self, *args, **kwargs):
        """Call from main thread."""
        payload = self.buf.get(*args, **kwargs)
        logger.debug("Removing RPC payload from ControlBuffer queue: %s", payload)
        return payload

class AgentConn(object):
    def __init__(self, env_status, cv, control_buffer, error_buffer, idle_timeout=None, exclusive=False):
        self.error_buffer = error_buffer

        self.env_status = env_status
        self.control_buffer = control_buffer
        self.cv = cv
        self.conns = {}
        self.exclusive = exclusive

        self.idle_timeout = idle_timeout
        self.last_disconnect_time = time.time()
        self._idle_message_interval = 10 # for logging

    def active_clients(self):
        return [conn for conn, stats in self.conns.items() if stats['active']]

    def listen(self, port=15900):
        logger.info('Starting Rewarder on port=%s', port)
        factory = websocket.WebSocketServerFactory()
        factory.agent_conn = self
        factory.protocol = RewarderProtocol

        reactor.callFromThread(reactor.listenTCP, port, factory)

    def check_message(self, conn):
        with self.cv:
            self.conns[conn]['messages'] += 1
            if self.conns[conn]['active']:
                return True
            elif self.conns[conn]['observer']:
                logger.info('CONNECTION STATUS: Marking connection as active: observer=%s peer=%s total_conns=%d', True, conn._request.peer, len(self.conns))
                self.conns[conn]['active'] = True
                return True
            else:
                # Note: if this exceptions, Autobahn will end up capturing
                # the errors since its hooks are called via
                # maybeDeferred. This won't always print the prettiest
                # stack trace.

                # This conn is neither active or an observer - before setting
                # active, let's see if there are any existing active,
                # non-observer conns.
                active = len([o for o in self.conns.values() if not o['observer'] and o['active']])
                if active > 0:
                    # Already full up, sorry!
                    logger.info('CONNECTION STATUS: Dropping new connection since already have %d non-observer conns (%d conns total)', active, len(self.conns))
                    # Sometimes connecting clients will time out before
                    # the connection is fully established, but the
                    # rewarder would still count the session as having
                    # come up. This, we try to wait until as late as
                    # possible to decide if a client is active.
                    conn.reject('The rewarder already has an active client. (HINT: if you obtained your environment through the allocator, make sure to run .configure(client_id=...) with a different client_id for each concurrent worker.)')
                    return False
                else:
                    logger.info('CONNECTION STATUS: Marking connection as active: observer=%s peer=%s total_conns=%d', False, conn._request.peer, len(self.conns))
                    self.conns[conn]['active'] = True
                    return True

    def _register(self, conn, observer=False):
        with self.cv:
            self.conns[conn] = {'messages': 0, 'observer': observer, 'active': False}
            self.cv.notifyAll()

    def _unregister(self, conn):
        with self.cv:
            try:
                stats = self.conns.pop(conn)
            except KeyError:
                stats = None
            else:
                self.cv.notifyAll()

            if stats is not None and stats['active']:
                self.last_disconnect_time = time.time()
                active = self.active_clients()
                logger.info('[%s] Active client disconnected (sent %d messages). Still have %d active clients left', utils.thread_name(), stats['messages'], len(active))
            else:
                logger.info('[%s] Non-active client disconnected', utils.thread_name())

            self.control_buffer.client_disconnect(conn, stats)

    def _broadcast(self, method, body, headers=None, conn=None):
        if conn:
            conns = [conn]
        else:
            conns = self.conns

        for conn in conns:
            conn.send_message(method, body, headers)

    def send_env_text(self, text, episode_id):
        ''' text channel to communicate with the agent '''
        reactor.callFromThread(self._send_env_text, text, episode_id)

    def _send_env_text(self, text, episode_id):
        self._broadcast('v0.env.text', {
            'text': text
        }, {'episode_id': episode_id})

    def send_env_observation(self, observation, episode_id):
        reactor.callFromThread(self._send_env_observation, observation, episode_id)

    def _send_env_observation(self, observation, episode_id, conn=None):
        self._broadcast('v0.env.observation', {
            'observation': observation,
        }, {'episode_id': episode_id}, conn=conn)

    def send_env_reward(self, reward, done, info, episode_id):
        pyprofile.incr('agent_conn.reward', reward)
        if done:
            pyprofile.incr('agent_conn.done')

        reactor.callFromThread(self._send_env_reward, reward, done, info, episode_id)

    def _send_env_reward(self, reward, done, info, episode_id):
        self._broadcast('v0.env.reward', {
            'reward': reward,
            'done': done,
            'info': info,
        }, {'episode_id': episode_id})

    def send_env_describe_from_env_info(self, env_info):
        assert env_info['fps'] is not None, "Missing fps: {}".format(env_info)
        self.send_env_describe(env_info['env_id'], env_info['env_state'], episode_id=env_info['episode_id'], fps=env_info['fps'])

    def send_env_describe(self, env_id, env_state, episode_id, fps, headers=None, parent_message_id=None, parent_context=None):
        reactor.callFromThread(self._send_env_describe, env_id, env_state, episode_id, fps, headers, parent_message_id, parent_context)

    def _send_env_describe(self, env_id, env_state, episode_id, fps, headers=None, parent_message_id=None, parent_context=None):
        conn = None
        if headers is None:
            headers = {}

        if parent_message_id is not None:
            headers['parent_message_id'] = parent_message_id
            headers['parent_runtime'] = time.time() - parent_context['start']
            conn = parent_context['conn']

        headers['episode_id'] = episode_id

        assert fps is not None
        # TODO: decide how to handle multiple concurrent envs
        self._broadcast('v0.env.describe', {
            'env_id': env_id,
            'env_state': env_state,
            'fps': fps,
        }, headers, conn)

    def send_reply_error(self, *args, **kwargs):
        reactor.callFromThread(self._send_reply_error, *args, **kwargs)

    def _send_reply_error(self, message, parent_message_id, parent_context):
        headers = {}
        headers['parent_message_id'] = parent_message_id
        headers['parent_runtime'] = time.time() - parent_context['start']
        conn = parent_context['conn']

        # TODO: decide how to handle multiple concurrent envs
        self._broadcast('v0.reply.error', {
            'message': message,
        }, headers, conn)

    def send_reply_env_reset(self, *args, **kwargs):
        reactor.callFromThread(self._send_reply_env_reset, *args, **kwargs)

    def _send_reply_env_reset(self, parent_message_id, parent_context, episode_id):
        headers = {}
        headers['parent_message_id'] = parent_message_id
        headers['parent_runtime'] = time.time() - parent_context['start']
        headers['episode_id'] = episode_id
        conn = parent_context['conn']

        # TODO: decide how to handle multiple concurrent envs
        self._broadcast('v0.reply.env.reset', {}, headers, conn)

    def check_status(self):
        with self.cv:
            if self.idle_timeout is None:
                return

            active = self.active_clients()
            if len(active) == 0:
                now = time.time()
                idle_duration = now - self.last_disconnect_time

                if self.idle_timeout is not None:
                    utils.periodic_log(
                        self, 'idle_timeout',
                        'No active clients for %.2fs (total client: %d); will exit due to idle timeout after %.0fs',
                        idle_duration, len(self.conns), self.idle_timeout, frequency=self._idle_message_interval)
                else:
                    utils.periodic_log(
                        self, 'idle_for',
                        'No active clients for %.2fs (total client: %d)',
                        idle_duration, len(self.conns), frequency=self._idle_message_interval)

                if self.idle_timeout is not None and idle_duration > self.idle_timeout:
                    self.error_buffer.record(Exit('EXIT CAUSE: idle timeout exceeded after {:.2f} seconds'.format(idle_duration)), wrap=False)

class RewardLogger(object):
    def __init__(self):
        self.reset(log=False, episode_stats=True)

    def reset(self, log=True, episode_stats=True):
        if log:
            self._log()

        self.last_print = time.time()
        # Could just maintain summary statistics, but we're not going
        # to have more than fps rewards at once, so it's fine to just
        # keep them all around.
        self.reward = []
        self.done = False
        self.info = {}
        self.count = 0

        if episode_stats:
            if log:
                self._log_reset()
            self.episode_reward = 0
            self.episode_count = 0
            self.episode_start = time.time()

    def record(self, reward, done, info):
        self.reward.append(reward)
        self.done = self.done or done
        self.info.update(info)
        self.count += 1

        self.episode_reward += reward
        self.episode_count += 1

        if time.time() - self.last_print > 1:
            self._log()
            self.reset(log=False, episode_stats=False)

    def _log_reset(self):
        logger.info('[%s] Ending previous episode: episode_reward=%s episode_count=%s episode_duration=%.2f', utils.thread_name(), self.episode_reward, self.episode_count, time.time() - self.episode_start)

    def _log(self):
        if 'rewarder.profile' in self.info:
            self.info['rewarder.profile'] = '<{} bytes>'.format(len(str(self.info['rewarder.profile'])))

        if len(self.reward) > 0:
            min_reward = min(self.reward)
            max_reward = max(self.reward)
        else:
            min_reward = '(empty)'
            max_reward = '(empty)'

        logger.info('[%s] Over past %.2fs, sent %s reward messages to agent: reward=%s reward_min=%s reward_max=%s done=%s info=%s',
                    utils.thread_name(), time.time() - self.last_print, self.count,
                    sum(self.reward), min_reward, max_reward,
                    self.done, self.info)
