import logging
from universe import pyprofile
import time
import ujson

from autobahn.twisted import websocket
from twisted.internet import defer

from universe import error

logger = logging.getLogger(__name__)
extra_logger = logging.getLogger('universe.extra.'+__name__)

class RemoteError(error.Error):
    pass

class RewarderClient(websocket.WebSocketClientProtocol):
    def __init__(self):
        super(RewarderClient, self).__init__()
        self._closed = False
        self._close_message = None

        self._connected = False
        self._requests = {}

        self._reset = None
        self._initial_reset = False

        self._connection_result = defer.Deferred()

    def send_reset(self, env_id, seed, fps, episode_id):
        self._initial_reset = True
        self._reset = {
            'env_id': env_id,
            'fps': fps,
            'episode_id': episode_id,
        }

        return self.send('v0.env.reset', {
            'seed': seed,
            'env_id': env_id,
            'fps': fps,
        }, {'episode_id': episode_id}, expect_reply=True)

    def _finish_reset(self, episode_id):
        extra_logger.info('[%s] Running finish_reset: %s', self.factory.label, episode_id)
        self.reward_buffer.reset(episode_id)

    def onConnect(self, request):
        self._message_id = 0
        self._requests = {}

        self.reward_buffer = self.factory.reward_buffer

        assert not self._connection_result.called
        self._connection_result.callback(self)
        self._connected = True

    def waitForWebsocketConnection(self):
        return self._connection_result

    def send(self, method, body, headers=None, expect_reply=False):
        if headers is None:
            headers = {}
        if self._closed:
            error_message = "Can't send message to closed connection"
            if self._close_message:
                error_message += ": {}".format(self._close_message)
            e = error.Error(error_message)
            if expect_reply:
                return defer.fail(e)
            else:
                raise e

        id = self._message_id

        self._message_id += 1
        new_headers = {
            'message_id': id,
            'sent_at': time.time(),
        }
        new_headers.update(headers)

        payload = {
            'method': method,
            'body': body,
            'headers': new_headers,
        }

        extra_logger.info('[%s] Sending message to rewarder: %s', self.factory.label, payload)
        self.sendMessage(ujson.dumps(payload).encode('utf-8'), False)

        if expect_reply:
            d = defer.Deferred()
            self._requests[id] = (payload, d)
            return d
        else:
            return None

    def _manual_recv(self, method, body, headers={}):
        """Used in the tests"""
        headers.setdefault('sent_at', time.time())
        return self.recv(self._make_context(), {'method': method, 'body': body, 'headers': headers})

    def recv(self, context, response):
        method = response['method']
        body = response['body']
        headers = response['headers']

        remote_time = headers['sent_at']
        local_time = context['start']

        episode_id = headers.get('episode_id')
        if episode_id is not None:
            self.reward_buffer.push_time(episode_id, remote_time, local_time)

        # Gets called by RewarderClient
        if method == 'v0.env.reward':
            episode_id = headers['episode_id']
            reward = body['reward']
            done = body['done']
            info = body['info']
            extra_logger.debug('[%s] Received %s: reward=%s done=%s info=%s episode_id=%s', self.factory.label, method, reward, done, info, episode_id)
            pyprofile.incr('rewarder_client.reward', reward)
            if done:
                pyprofile.incr('rewarder_client.done')
            self.reward_buffer.push(episode_id, reward, done, info)
        elif method == 'v0.env.text':
            episode_id = headers['episode_id']
            text = body['text']
            extra_logger.debug('[%s] Received %s: text=%s episode_id=%s', self.factory.label, method, text, episode_id)
            self.reward_buffer.push_text(episode_id, text)
        elif method == 'v0.env.observation':
            episode_id = headers['episode_id']
            jsonable = body['observation']
            extra_logger.debug('[%s] Received %s: observation=%s episode_id=%s', self.factory.label, method, jsonable, episode_id)
            self.reward_buffer.set_observation(episode_id=episode_id, observation=jsonable)
        elif method == 'v0.env.describe':
            episode_id = headers['episode_id']
            env_id = body['env_id']
            env_state = body['env_state']
            fps = body['fps']
            extra_logger.info('[%s] Received %s: env_id=%s env_state=%s episode_id=%s',
                              self.factory.label, method, env_id, env_state, episode_id)
            self.reward_buffer.set_env_info(env_state, env_id=env_id, episode_id=episode_id, fps=fps)
        elif method == 'v0.reply.env.reset':
            episode_id = headers['episode_id']
            self._finish_reset(episode_id)
        elif method in ['v0.reply.error', 'v0.reply.control.ping']:
            assert headers.get('parent_message_id') is not None
        elif method == 'v0.connection.close':
            assert headers.get('parent_message_id') is None
            logger.debug('Server hanging up: %s', body['message'])

            self._close_message = body['message']
            e = error.Error(body['message'])
            self.factory.record_error(e)
        else:
            logger.error('Unrecognized websocket method: method=%s body=%s headers=%s (consider adding to rewarder_state.py)', method, body, headers)
            return

        parent_id = headers.get('parent_message_id')
        if parent_id is not None:
            try:
                spec = self._requests.pop(parent_id)
            except KeyError:
                logger.error('[%s] Received extra reply to %d; ignoring: method=%s body=%s headers=%s ', self.factory.label, parent_id, method, body, headers)
            else:
                request, d = spec
                if method != 'v0.reply.error':
                    d.callback((context, request, response))
                else:
                    e = RemoteError('[{}] Remote error: {}'.format(self.factory.label, body['message']))
                    d.errback(e)

    def _make_context(self):
        return {'start': time.time()}

    def onMessage(self, payload, isBinary):
        extra_logger.debug('[%s] Received payload: %s', self.factory.label, payload)
        assert not isBinary
        payload = ujson.loads(payload)

        context = self._make_context()
        latency = context['start'] - payload['headers']['sent_at']
        pyprofile.incr('rewarder_protocol.messages')
        pyprofile.incr('rewarder_protocol.messages.{}'.format(payload['method']))

        # Double latency to model RTT
        pyprofile.timing('rewarder_protocol.latency.rtt.skew_unadjusted', 2*latency)
        if latency < 0:
            pyprofile.incr('rewarder_protocol.latency.rtt.skew_unadjusted.negative')

        self.recv(context, payload)

    def onClose(self, wasClean, code, reason):
        if self._close_message:
            return

        if not self._connected:
            assert not self._connection_result.called
            self._connection_result.errback(error.ConnectionError(reason))
            return

        if not self._closed:
            error_message = 'Lost connection: {} (clean={} code={})'.format(reason, wasClean, code)
            reason = error.Error(error_message)
            # TODO: it's not an error if we requested it
            self.factory.record_error(reason)
        else:
            reason = error.Error("We closed the connection: {}".format(reason))

        for request, d in self._requests.values():
            d.errback(reason)

    def close(self, code=1000, reason=None):
        self._closed = True
        extra_logger.info('[%s] Client closing websocket connection because of call to close(code=%s, reason=%s)', self.factory.label, code, reason)
        self.sendClose(code, reason)
        self.transport.loseConnection()
