import logging

from twisted.internet import defer, endpoints

from universe import error, utils
from universe.twisty import reactor
from universe.vncdriver import screen, vnc_client

logger = logging.getLogger(__name__)

class VNCSession(object):
    def __init__(self, remotes, error_buffer):
        self.remotes = remotes
        self.error_buffer = error_buffer
        self._pyglet_screen = None
        self.connect()

    def connect(self):
        utils.blockingCallFromThread(self._connect)

    def _connect(self):
        deferreds = []

        for i, remote in enumerate(self.remotes):
            d = defer.Deferred()
            deferreds.append(d)

            factory = vnc_client.client_factory(d, self.error_buffer)
            factory.rewarder_session = self
            factory.label = 'vnc:{}:{}'.format(i, remote)
            endpoint = endpoints.clientFromString(reactor, 'tcp:'+remote)

            def success(i):
                logger.info('[%s] VNC connection established', factory.label)

            def fail(reason):
                reason = error.Error('[{}] Connection failed: {}'.format(factory.label, reason.value))
                try:
                    d.errback(utils.format_error(reason))
                except defer.AlreadyCalledError:
                    pass
            endpoint.connect(factory).addCallback(success).addErrback(fail)

        d = defer.DeferredList(deferreds, fireOnOneErrback=True)

        def success(results):
            # Store the _clients list when connected
            self._clients = [client for success, client in results]
        d.addCallback(success)
        return d

    def flip(self):
        observation_n = []
        info_n = []
        for i, client in enumerate(self._clients):
            observation, info = client.numpy_screen.flip()
            updates = info['vnc_session.framebuffer_updates']

            # Keep the pyglet screen fed, but don't flip it until the user calls render
            if i == 0 and self._pyglet_screen:
                for update in updates:
                    self._pyglet_screen.apply(update)

            observation_n.append(observation)
            info_n.append({'vnc.updates.n': len(updates)})

        return observation_n, info_n

    def peek(self):
        observations = [client.numpy_screen.peek() for client in self._clients]
        return observations

    def step(self, action):
        reactor.callFromThread(self._step, action)
        return self.flip()

    def _step(self, action):
        try:
            for a, client in zip(action, self._clients):
                for event in a:
                    if event[0] == 'KeyEvent':
                        key, down = event[1:]
                        client.send_KeyEvent(key, down)
                    elif event[0] == 'PointerEvent':
                        x, y, buttomask = event[1:]
                        client.send_PointerEvent(x, y, buttomask)
                    else:
                        raise error.Error('Bad event type: {}'.format(type))
        except Exception as e:
            self.error_buffer.record(e)

    def render(self):
        if not self._pyglet_screen:
            start = self.peek()[0]
            self._pyglet_screen = screen.PygletScreen(start)
        self._pyglet_screen.flip()

    def close(self):
        utils.blockingCallFromThread(self._close)

    def _close(self):
        if getattr(self, '_clients', None) is not None:
            for client in self._clients:
                client.close()
            self._clients = None
