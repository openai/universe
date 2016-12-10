import os
import re
import signal
import time

from universe import error
from universe.twisty import reactor
from twisted.internet import defer, protocol
import twisted.internet.error
import logging

logger = logging.getLogger(__name__)
extra_logger = logging.getLogger('universe.extra.'+__name__)

class ConnectionTimer(protocol.Protocol):
    def connectionMade(self):
        self.transport.loseConnection()

def start(endpoint):
    start = time.time()
    return endpoint.connect(
        protocol.ClientFactory.forProtocol(ConnectionTimer)
    ).addCallback(lambda _: time.time() - start)

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
