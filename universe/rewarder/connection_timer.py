import os
import random
import re
import signal
import time

from universe import error
from universe.twisty import reactor
from twisted.internet import defer, protocol, task
import twisted.internet.error
import logging

logger = logging.getLogger(__name__)
extra_logger = logging.getLogger('universe.extra.'+__name__)

class ConnectionTimer(protocol.Protocol):
    def connectionMade(self):
        self.transport.loseConnection()

def connection_timer_factory():
    factory = protocol.ClientFactory()
    factory.protocol = ConnectionTimer
    return factory

class StopWatch(object):
    def start(self):
        self.start_time = time.time()

    def stop(self):
        return time.time() - self.start_time

# TODO: clean this up
def start(endpoint, max_attempts=0):
    # Use an object for timing so that we can mutate it within the closure
    stop_watch = StopWatch()

    def success(client):
        return stop_watch.stop()

    def error(failure, retry):
        # some websocket implementations (like websocketcpp) can fail when connections are lost too quickly
        if retry == 0:
            raise ConnectionTimerException("Max retries")
        backoff = 2 ** (max_attempts - retry + 1) + random.randint(42, 100)
        logger.info('Throttling down websocket creation after connection error (this is normal) - waiting %dms - '
                    'error details %s', backoff, error)
        d = task.deferLater(reactor, backoff / 1000., go, retry - 1)
        return d

    def go(retry):
        stop_watch.start()
        factory = connection_timer_factory()
        d = endpoint.connect(factory)
        d.addCallback(success)
        d.addErrback(error, retry)
        return d

    return go(retry=max_attempts)

def measure_clock_skew(label, host):
    cmd = ['ntpdate', '-q', '-p', '8', host]
    extra_logger.info('[%s] Starting network calibration with %s', label, ' '.join(cmd))
    skew = Clockskew(label, cmd)
    # TODO: search PATH for this?
    process = reactor.spawnProcess(skew, '/usr/sbin/ntpdate', cmd, {})
    # process = reactor.spawnProcess(skew, '/bin/sleep', ['sleep', '2'], {})

    t = float(os.environ.get('UNIVERSE_NTPDATE_TIMEOUT', 20))
    def timeout():
        if process.pid:
            logger.error('[%s] %s call timed out after %ss; killing the subprocess. This is ok, but you could have more accurate timings by enabling UDP port 123 traffic to your env. (Alternatively, you can try increasing the timeout by setting environment variable UNIVERSE_NTPDATE_TIMEOUT=10.)', label, ' '.join(cmd), t)
            process.signalProcess(signal.SIGKILL)
            process.reapProcess()
    # TODO: make this part of the connection string
    reactor.callLater(t, timeout)
    return skew.deferred

class Clockskew(protocol.ProcessProtocol):
    def __init__(self, label, cmd):
        self.label = label
        self._cmd = cmd

        self.deferred = defer.Deferred()
        self.out = []
        self.err = []

    def outReceived(self, data):
        self.out.append(data)

    def errReceived(self, data):
        self.err.append(data)

    def processExited(self, reason):
        if isinstance(reason.value, twisted.internet.error.ProcessDone):
            out = b''.join(self.out).decode('utf-8')
            match = re.search('offset ([\d.-]+) sec', out)
            if match is not None:
                offset = float(match.group(1))
                self.deferred.callback(offset)
            else:
                self.deferred.errback(error.Error('Could not parse offset: %s', out))
        else:
            err = b''.join(self.err)
            self.deferred.errback(error.Error('{} failed with status {}: stderr={!r}'.format(self._cmd, reason.value.exitCode, err)))

class ConnectionTimerException(Exception):
    pass
