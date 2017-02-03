# A simple in-memory stats library
#
# Inspired by statsd: http://statsd.readthedocs.io/en/v3.1/types.html#gauges
import collections
import json
import logging
import numbers
import numpy as np
import os
import six
import threading
import time

logger = logging.getLogger(__name__)

BYTES = 'bytes'
SECONDS = 'seconds'

class Error(Exception):
    pass

class ExponentialAverage(object):
    def __init__(self, decay=0.1):
        self.decay = decay
        self.last_update = None
        self.last_data_decay = None
        self._avg = None

    def add(self, data):
        assert isinstance(data, numbers.Number)
        if self.last_update is None:
            self._avg = data
            self.last_update = time.time()
            self.last_data_decay = 1
        else:
            now = time.time()
            delta = now - self.last_update
            if delta < 0:
                # Time is allowed to go a little backwards (NTP update, etc)
                logger.warn("Backwards delta value: {}".format(delta))
                # Treat this entry as if it happened with 0 delta
                delta = 0
            if delta != 0:
                self.last_data_decay = (1 - self.decay**delta) * 1/delta
                self._avg = self.decay**delta * self._avg + self.last_data_decay * data
            else:
                # Don't divide by zero; just reuse the last delta. Should stack well
                self._avg += self.last_data_decay * data
            self.last_update = now

    def avg(self):
        return self._avg

class RunningVariance(object):
    """ Implements Welford's algorithm for computing a running mean
    and standard deviation as described at:
        http://www.johndcook.com/standard_deviation.html
    can take single values or iterables
    Properties:
        mean    - returns the mean
        std     - returns the std
        meanfull- returns the mean and std of the mean
    Usage:
        >>> foo = Welford()
        >>> foo(range(100))
        >>> foo
        <Welford: 49.5 +- 29.0114919759>
        >>> foo([1]*1000)
        >>> foo
        <Welford: 5.40909090909 +- 16.4437417146>
        >>> foo.mean
        5.409090909090906
        >>> foo.std
        16.44374171455467
        >>> foo.meanfull
        (5.409090909090906, 0.4957974674244838)
    """

    def __init__(self):
        self.k = 0
        self.M = 0
        self.S = 0

    def add(self,x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M)*1./self.k
        newS = self.S + (x - self.M)*(x - newM)
        self.M, self.S = newM, newS

    def mean(self):
        return self.M
    def meanfull(self):
        return self.mean, self.std/np.sqrt(self.k)
    def std(self):
        if self.k==1:
            return 0
        return np.sqrt(self.S/(self.k-1))
    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)

def pretty(d, unit):
    if unit is None:
        return d
    elif unit == BYTES:
        return pretty_bytes(d)
    elif unit == SECONDS:
        return pretty_seconds(d)
    else:
        raise Error('No such unit: {}'.format(unit))

def pretty_bytes(b):
    if b is None:
        return None

    assert isinstance(b, numbers.Number), "Surprising type for data: {} ({!r})".format(type(b), b)
    if b > 1000 * 1000:
        return '{:.0f}MB'.format(b/1000.0/1000.0)
    elif b > 1000:
        return '{:.0f}kB'.format(b/1000.0)
    else:
        return '{:.0f}B'.format(b)

def pretty_seconds(t):
    a_t = abs(t)
    if a_t < 0.001:
        return '{:.2f}us'.format(1000*1000*t)
    elif a_t < 1:
        return '{:.2f}ms'.format(1000*t)
    else:
        return '{:.2f}s'.format(t)

def thread_id():
    return threading.current_thread().ident

class StackProfile(object):
    def __init__(self, profile):
        self.profile = profile

        self.stack_by_thread = {}
        self.lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.pop()

    def push(self, event):
        stack = self._current_stack()
        stack.append({
            'name': event,
            'start': time.time(),
        })
        return self

    def pop(self):
        stack = self._current_stack()
        event = stack.pop()
        name = event['name']
        start = event['start']

        with self.profile as txn:
            delta = time.time() - start
            txn.timing(name, delta)
            # These are now subsumed by the timers key
            # txn.incr(name + '.total_time', delta, unit=SECONDS)
            # txn.incr(name + '.calls')

    def _current_stack(self):
        id = thread_id()

        try:
            stack = self.stack_by_thread[id]
        except KeyError:
            with self.lock:
                # Only current thread should be adding to this entry anyway
                assert id not in self.stack_by_thread
                stack = self.stack_by_thread[id] = []
        return stack

class Profile(object):
    def __init__(self, print_frequency=None, print_filter=None):
        if print_filter is None:
            print_filter = lambda event: True

        self.lock = threading.RLock()

        self.print_frequency = print_frequency
        self.last_export = None
        self.export_hooks = [self._print_export]

        self.print_filter = print_filter
        self._in_txn = False

        self.reset()

    def reset(self):
        self.timers = {}
        self.counters = {}
        self.gauges = {}

    def add_export_hook(self, hook):
        self.export_hooks.append(hook)

    def __enter__(self):
        self.lock.acquire()
        self._in_txn = True
        return self

    def __exit__(self, type, value, tb):
        self._in_txn = False
        self._print_if_needed()
        self.lock.release()

    def timing(self, event, time):
        assert isinstance(event, six.string_types)
        # return
        with self.lock:
            if event not in self.timers:
                self.timers[event] = {
                    'total': 0,
                    'calls': 0,
                    'std': RunningVariance(),
                }
            self.timers[event]['total'] += time
            self.timers[event]['calls'] += 1
            self.timers[event]['std'].add(time)

            self._print_if_needed()

    def incr(self, event, amount=1, unit=None):
        assert isinstance(event, six.string_types)
        # return
        with self.lock:
            if event not in self.counters:
                self.counters[event] = {
                    'total': 0,
                    'calls': 0,
                    'rate': ExponentialAverage(),
                    'unit': unit,
                    'std': RunningVariance(),
                }
            self.counters[event]['total'] += amount
            self.counters[event]['calls'] += 1
            self.counters[event]['rate'].add(amount)
            self.counters[event]['std'].add(amount)

            self._print_if_needed()

    def gauge(self, event, value, delta=False, unit=None):
        assert isinstance(event, six.string_types)
        with self.lock:
            if event not in self.gauges:
                self.gauges[event] = {
                    'value': 0,
                    'calls': 0,
                    'unit': unit,
                    'std': RunningVariance(),
                }
            if delta:
                self.gauges[event]['value'] += value
            else:
                self.gauges[event]['value'] = value
            self.gauges[event]['calls'] += 1
            self.gauges[event]['std'].add(value)

            self._print_if_needed()

    def _print_if_needed(self):
        """Assumes you hold the lock"""
        if self._in_txn or self.print_frequency is None:
            return
        elif self.last_export is not None and \
             self.last_export + self.print_frequency > time.time():
            return

        self.export()

    def export(self, log=True, reset=True):
        with self.lock:
            if self.last_export is None:
                self.last_export = time.time()
            delta = time.time() - self.last_export
            self.last_export = time.time()

            timers = {}
            for event, stat in self.timers.items():
                timers[event] = {
                    'mean': stat['std'].mean(),
                    'std': stat['std'].std(),
                    'calls': stat['calls'],
                    'unit': 'seconds',
                }

            counters = {}
            for counter, stat in self.counters.items():
                counters[counter] = {
                    'calls': stat['calls'],
                    'std': stat['std'].std(),
                    'mean': stat['std'].mean(),
                    'unit': stat['unit'],
                    'total': stat['total'],
                    'rate': stat['rate'].avg(),
                }

            gauges = {}
            for gauge, stat in self.gauges.items():
                gauges[gauge] = {
                    'value': stat['value'],
                    'calls': stat['calls'],
                    'std': stat['std'].std(),
                    'mean': stat['std'].mean(),
                    'unit': stat['unit'],
                }

            export = {
                'timers': timers,
                'counters': counters,
                'gauges': gauges,
                'metadata': {
                    'period': delta,
                }
            }
            if log:
                for hook in self.export_hooks:
                    hook(export)
            if reset:
                self.reset()
            return export

    def _print_export(self, export):
        timers = {}
        for event, stat in sorted(export['timers'].items()):
            if not self.print_filter(event):
                continue

            timers[event] = {
                'mean': pretty_seconds(stat['mean']),
                'std': pretty_seconds(stat['std']),
                'calls': stat['calls'],
            }

        counters = collections.OrderedDict({})
        for counter, stat in sorted(export['counters'].items()):
            if not self.print_filter(counter):
                continue

            unit = stat['unit']
            counters[counter] = {
                'calls': stat['calls'],
                'std': pretty(stat['std'], unit),
                'mean': pretty(stat['mean'], unit),
            }

        gauges = collections.OrderedDict({})
        for gauge, stat in sorted(export['gauges'].items()):
            if not self.print_filter(gauge):
                continue

            unit = stat['unit']
            gauges[gauge] = {
                'value': pretty(stat['value'], unit),
                'calls': stat['calls'],
                'std': pretty(stat['std'], unit),
                'mean': pretty(stat['mean'], unit),
            }

        # A bit of a hack, but we want this time to be as inclusive as
        # possible.
        export['metadata']['export_time'] = time.time() - self.last_export

        # We do the explicit OrderedDict and json.dumps to order
        # keys. Maybe there's a better way?
        logger.info('[pyprofile] period=%s timers=%s counters=%s gauges=%s (export_time=%s)',
                    pretty_seconds(export['metadata']['period']),
                    json.dumps(timers), json.dumps(counters), json.dumps(gauges),
                    pretty_seconds(export['metadata']['export_time']),
        )

print_frequency = os.environ.get('PYPROFILE_FREQUENCY')
if print_frequency is not None:
    print_frequency = float(print_frequency)

print_prefix = os.environ.get('PYPROFILE_PREFIX')
if print_prefix is not None:
    print_filter = lambda event: event.startswith(print_prefix)
else:
    print_filter = None

profile = Profile(print_frequency=print_frequency, print_filter=print_filter)
stack_profile = StackProfile(profile)

push = stack_profile.push
pop = stack_profile.pop
incr = profile.incr
timing = profile.timing
gauge = profile.gauge
export = profile.export
