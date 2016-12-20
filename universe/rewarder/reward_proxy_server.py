import json
import logging
import os
import time

from autobahn.twisted import websocket
from universe.twisty import reactor
from twisted.internet import endpoints

logger = logging.getLogger(__name__)

class RewardServerClient(websocket.WebSocketClientProtocol, object):
    def __init__(self):
        super(RewardServerClient, self).__init__()
        self.id = -1

        self.proxy_server = None
        self._connected = False

    def onConnect(self, request):
        self.id = self.factory.proxy_server.id
        logger.info('[RewardProxyClient] [%d] Connected to rewarder', self.id)
        self.proxy_server = self.factory.proxy_server
        self._connected = True

        buffered = self.proxy_server.pop_buffer()
        logger.info('[RewardProxyClient] [%d] Flushing %d buffered messages', self.id, len(buffered))
        for msg in buffered:
            self.sendMessage(msg)

    def onOpen(self):
        logger.info('[RewardProxyClient] [%d] Rewarder websocket connection established', self.id)

    def onMessage(self, msg, isBinary):
        logger.debug('[RewardProxyClient] [%d] Received message from server: %s', self.id, msg)
        self.proxy_server.sendMessage(msg)

        # Record the message
        self.proxy_server.record_message(msg.decode('utf-8'), from_rewarder=True)

        # # Process the message for recording
        # method, headers, body = unpack_message(msg)
        #
        # if method == "env.reward":
        #     # {"body":{"info":{"episode":0},"reward":0.0,"done":false},
        #     # "headers":{"sent_at":1473126129.231828928,"message_id":207},
        #     # "method":"env.reward"}

    def onClose(self, wasClean, code, reason):
        logger.info('[RewardProxyClient] [%d] Rewarder websocket connection closed: %s', self.id, reason)

    def close(self):
        logger.info('[RewardProxyClient] [%d] Closing connection', self.id)
        self.transport.loseConnection()


class RewardProxyServer(websocket.WebSocketServerProtocol, object):
    _next_id = 0
    _n_open_files = 0

    @classmethod
    def next_id(cls):
        id = cls._next_id
        cls._next_id += 1
        return id

    def __init__(self):
        super(RewardProxyServer, self).__init__()
        self.id = self.next_id()
        self.client = None
        self.file = None  # We do not open open the file until we have established an end-to-end connection
        self.buffered = []

        self._closed = False

    def pop_buffer(self):
        """Called by the client once it's ready to start sending messages.
        """
        buffered = self.buffered
        self.buffered = []
        return buffered

    def begin_recording(self):
        """
        Open the file and write the metadata header to describe this recording. Called after we establish an end-to-end connection
        This uses Version 1 of our protocol

        Version 0 can be seen here: https://github.com/openai/universe/blob/f85a7779c3847fa86ec7bb513a1da0d3158dda78/bin/recording_agent.py
        """
        logger.info("[RewardProxyServer] [%d] Starting recording", self.id)

        if self._closed:
            logger.error(
                "[RewardProxyServer] [%d] Attempted to start writing although client connection is already closed. Aborting", self.id)
            self.close()
            return

        if self._n_open_files != 0:
            logger.error("[RewardProxyServer] [%d] WARNING: n open rewards files = %s. This is unexpected. Dropping connection.", self.id, self._n_open_files)
            self.close()
            return

        logfile_path = os.path.join(self.factory.logfile_dir, 'rewards.demo')
        logger.info('Recording to {}'.format(logfile_path))
        self.file = open(logfile_path, 'w')

        self._n_open_files += 1
        logger.info("[RewardProxyServer] [%d] n open rewards files incremented: %s", self.id, self._n_open_files)

        self.file.write(json.dumps({
            'version': 1,
            '_debug_version': '0.0.1',  # Give this an internal version for debugging corrupt reward.demo files # TODO, pull this from setup.py or the host docker image
        }))
        self.file.write('\n')
        self.file.flush()

        logger.info("[RewardProxyServer] [%d] Wrote version number", self.id)

    def onConnect(self, request):
        logger.info('[RewardProxyServer] [%d] Client connecting: %s', self.id, request.peer)
        self._request = request

    def onOpen(self):
        logger.info("[RewardProxyServer] [%d] Websocket connection established", self.id)
        self.connect_upstream()

    def connect_upstream(self, tries=1, max_attempts=7):
        if self._closed:
            logger.info("[RewardProxyServer] [%d] Attempted to connect upstream although client connection is already closed. Aborting",
                        self.id)
            return

        remote = getattr(self.factory, 'rewarder_address', 'localhost:15900')
        endpoint = endpoints.clientFromString(reactor, 'tcp:' + remote)
        client_factory = websocket.WebSocketClientFactory('ws://' + remote)
        headers = {'authorization': self._request.headers['authorization']}
        if self._request.headers.get('openai-observer'):
            headers['openai-observer'] = self._request.headers.get('openai-observer')
        client_factory.headers = headers
        client_factory.protocol = RewardServerClient
        client_factory.proxy_server = self
        client_factory.endpoint = endpoint

        logger.info("[RewardProxyServer] [%d] Connecting to upstream %s (try %d/%d)", self.id, remote, tries, max_attempts)

        def _connect_callback(client):
            logger.info('[RewardProxyServer] [%d] Upstream connection %s established', self.id, remote)
            self.client = client
            if self.factory.logfile_dir:
                self.begin_recording()

        def _connect_errback(reason):
            if tries < max_attempts:
                # Somewhat arbitrary exponential backoff: should be
                # pretty rare, and indicate that we're just starting
                # up.
                delay = 1.5 ** tries
                logger.info('[RewardProxyServer] [%d] Connection to %s failed: %s. Try %d/%d; going to retry in %fs', self.id, remote, reason, tries, max_attempts, delay)
                reactor.callLater(
                    delay, self.connect_upstream,
                    tries=tries+1, max_attempts=max_attempts)
            else:
                logger.error('[RewardProxyServer] [%d] Connection to %s failed: %s. Completed %d/%d atttempts; disconnecting.', self.id, remote, reason, tries, max_attempts)
                self.transport.loseConnection()

        endpoint.connect(client_factory).addCallbacks(_connect_callback, _connect_errback)

    def close(self):
        logger.info('[RewardProxyServer] [%d] Closing...', self.id)
        self.transport.loseConnection()

    def onClose(self, wasClean, code, reason):
        logger.info('[RewardProxyServer] [%d] Client connection closed: %s', self.id, reason)
        if self.client:
            self.client.close()
        if self.file:
            self.file.close()

        self._closed = True

    def onMessage(self, msg, binary):
        logger.debug('[RewardProxyServer] [%d] Received message from client: %s', self.id, msg)

        # Pass the message on to the client
        if self.client and self.client._connected:
            self.client.sendMessage(msg)
        else:
            self.buffered.append(msg)

        self.record_message(msg.decode('utf-8'), from_rewarder=False)

    def record_message(self, msg, from_rewarder):
        """Record a message to our rewards.demo file if it is has been opened"""
        if self.file:
            # Include an authoritative timestamp (because the `sent_at` from the server is likely to be different
            timestamped_message = {
                'timestamp': time.time(),
                'message': json.loads(msg),
                'from_rewarder': from_rewarder,
            }
            self.file.write(json.dumps(timestamped_message))
            self.file.write('\n')
            self.file.flush()
