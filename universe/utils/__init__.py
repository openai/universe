import logging
import six
import sys
if six.PY2:
    import Queue as queue
else:
    import queue
import threading
import signal
from twisted.internet import defer

from universe.twisty import reactor

logger = logging.getLogger(__name__)

class ErrorBuffer(object):
    def __init__(self):
        self.queue = queue.Queue()

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        if value is not None:
            self.record(value)

    def __call__(self, error, wrap=True):
        self.record(error, wrap=True)

    def record(self, error, wrap=True):
        logger.debug('Error in thread %s: %s', threading.current_thread().name, error)
        if wrap:
            error = format_error(error)

        try:
            self.queue.put_nowait(error)
        except queue.Full:
            pass

    def check(self, timeout=None):
        if timeout is None:
            timeout = 0

        try:
            error = self.queue.get(timeout=timeout)
        except queue.Empty:
            return
        else:
            raise error

    def blocking_check(self, timeout=None):
        # TODO: get rid of this method
        if timeout is None:
            while True:
                self.check(timeout=3600)
        else:
            self.check(timeout)


from twisted.python import failure
import traceback
import threading
from universe import error
def format_error(e):
    # errback automatically wraps everything in a Twisted Failure
    if isinstance(e, failure.Failure):
        e = e.value

    if isinstance(e, str):
        err_string = e
    elif six.PY2:
        err_string = traceback.format_exc(e).rstrip()
    else:
        err_string = ''.join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip()

    if err_string == 'None':
        # Reasonable heuristic for exceptions that were created by hand
        last = traceback.format_stack()[-2]
        err_string = '{}\n  {}'.format(e, last)
    # Quick and dirty hack for now.
    err_string = err_string.replace('Connection to the other side was lost in a non-clean fashion', 'Connection to the other side was lost in a non-clean fashion (HINT: this generally actually means we got a connection refused error. Check that the remote is actually running.)')
    return error.Error(err_string)

def queue_get(local_queue):
    while True:
        try:
            result = local_queue.get(timeout=1000)
        except queue.Empty:
            pass
        else:
            return result

def blockingCallFromThread(f, *a, **kw):
    local_queue = queue.Queue()
    def _callFromThread():
        result = defer.maybeDeferred(f, *a, **kw)
        result.addBoth(local_queue.put)
    reactor.callFromThread(_callFromThread)
    result = queue_get(local_queue)
    if isinstance(result, failure.Failure):
        if result.frames:
            e = error.Error(str(result))
        else:
            e = result.value
        raise e
    return result

from gym import spaces
def repeat_space(space, n):
    return spaces.Tuple([space] * n)

import base64
import uuid
def random_alphanumeric(length=14):
    buf = []
    while len(buf) < length:
        entropy = base64.encodestring(uuid.uuid4().bytes).decode('ascii')
        bytes = [c for c in entropy if c.isalnum()]
        buf += bytes
    return ''.join(buf)[:length]


def best_effort(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except:
        if six.PY2:
            logging.error('Error in %s:', function.__name__)
            traceback.print_exc()
        else:
            logging.error('Error in %s:', function.__name__)
            logger.error(traceback.format_exc())
        return None

import base64
def basic_auth_encode(username, password=''):
    fmt = '{}:{}'.format(username, password)
    return 'Basic ' + base64.encodestring(fmt.encode('utf-8')).rstrip().decode('utf-8')

def basic_auth_decode(header):
    if header.startswith('Basic '):
        header = header[len('Basic '):]
        decoded = base64.decodestring(header.encode('utf-8')).decode('utf-8')
        username, password = decoded.split(':')
        return username, password
    else:
        return None

import os
def default_password():
    if os.path.exists('/usr/local/openai/privileged_state/password'):
        with open('/usr/local/openai/privileged_state/password') as f:
            return f.read().strip()
    return 'openai'

import logging
import time
logger = logging.getLogger(__name__)
class PeriodicLog(object):
    def log(self, obj, name, msg, *args, **kwargs):
        try:
            info = obj._periodic_log_info
        except AttributeError:
            info = obj._periodic_log_info = {}

        # Would be better to use a frequency=... arg after kwargs, but
        # that isn't py2 compatible.
        frequency = kwargs.pop('frequency', 1)
        delay = kwargs.pop('delay', 0)
        last_log = info.setdefault(name, time.time()-frequency+delay)
        if time.time() - last_log < frequency:
            return
        info[name] = time.time()
        logger.info('[{}] {}'.format(name, msg), *args)

    def log_debug(self, obj, name, msg, *args, **kwargs):
        try:
            info = obj._periodic_log_debug
        except AttributeError:
            info = obj._periodic_log_debug = {}

        frequency = kwargs.pop('frequency', 1)
        delay = kwargs.pop('delay', 0)
        last_log = info.setdefault(name, time.time()-frequency+delay)
        if time.time() - last_log < frequency:
            return
        info[name] = time.time()
        logger.debug('[{}] {}'.format(name, msg), *args)

_periodic = PeriodicLog()
periodic_log = _periodic.log
periodic_log_debug = _periodic.log_debug

import threading
def thread_name():
    return threading.current_thread().name

def exit_on_signal():
    """
    Install a signal handler for HUP, INT, and TERM to call exit, allowing clean shutdown.
    When running a universe environment, it's important to shut down the container when the
    agent dies so you should either call this or otherwise arrange to exit on signals.
    """
    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
