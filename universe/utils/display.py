# -*- coding: utf-8 -*-
import logging

import six
import numpy as np

logger = logging.getLogger(__name__)

# We log these with logger, which in py2 chokes on unicode
def fmt_plusminus(mean, dev):
    if six.PY3:
        return mean + '±' + dev
    else:
        # Logging unicode in py2 is asking for trouble
        return mean + '+-' + dev

def compute_timestamps_pair_max(time_m_2, flat=True):
    if flat:
        # Ignore empty inputs, which happens when environments are resetting.
        time_m_2 = [[x for x in time_m_2 if len(x)]]

    if len(time_m_2) == 0:
        return None, None

    # We concatenate the (min, max) lags from a variety of runs. Those
    # runs may have different lengths.
    time_m_2 = [np.array(m) for m in time_m_2]

    timestamp_m = []
    data_m = []
    for m in time_m_2:
        if len(m) > 0:
            timestamp, data = compute_timestamps_sigma(m[:, 1])
            timestamp_m.append(timestamp)
            data_m.append(data)
        else:
            timestamp_m.append(None)
            data_m.append({})
    return timestamp_m, data_m

def display_timestamps_pair_compact(time_m_2):
    """Takes a list of the following form: [(a1, b1), (a2, b2), ...] and
    returns a string a_mean-b_mean, flooring out at 0.
    """
    if len(time_m_2) == 0:
        return '(empty)'

    time_m_2 = np.array(time_m_2)

    low = time_m_2[:, 0].mean()
    high = time_m_2[:, 1].mean()

    low = max(low, 0)

    # Not sure if this'll always be true, and not worth crashing over
    if high < 0:
        logger.warn('Harmless warning: upper-bound on clock skew is negative: (%s, %s). Please let Greg know about this.', low, high)

    return '{}-{}'.format(display_timestamp(low), display_timestamp(high))

def display_timestamps_pair(time_m_2):
    """Takes a list of the following form: [(a1, b1), (a2, b2), ...] and
    returns a string (a_mean+/-a_error, b_mean+/-b_error).
    """
    if len(time_m_2) == 0:
        return '(empty)'

    time_m_2 = np.array(time_m_2)
    return '({}, {})'.format(
        display_timestamps(time_m_2[:, 0]),
        display_timestamps(time_m_2[:, 1]),
    )

def compute_timestamps_sigma_n(time_m):
    timestamp_m = []
    data_m = []

    for t in time_m:
        timestamp, data = compute_timestamps(t)
        timestamp_m.append(timestamp)
        data_m.append(data)

    return timestamp_m, data_m

def compute_timestamps_sigma(time_m):
    if len(time_m) == 0:
        return None, {}

    mean = np.mean(time_m)
    std = standard_error(time_m)
    scale, units = pick_time_units(mean)
    return fmt_plusminus('{:.2f}{}'.format(mean * scale, units), '{:.2f}{}'.format(std * scale, units)), {'mean': mean}

def display_timestamps(time_m):
    res, _ = compute_timestamps(time_m)
    if res is None:
        return '(empty)'
    else:
        return res

def compute_timestamps(time_m):
    if len(time_m) == 0:
        return None, {}

    mean = np.mean(time_m)
    std = standard_error(time_m)
    return fmt_plusminus(display_timestamp(mean), display_timestamp(std)), {'mean': mean}

def display_timestamps_n(time_m):
    # concatenate all the n's timesteps together, then display_timestamps on it
    return display_timestamps(np.concatenate(time_m))

def standard_error(ary, axis=0):
    if len(ary) > 1:
        return np.std(ary, axis=axis) / np.sqrt(len(ary) - 1)
    else:
        return np.std(ary, axis=axis)

def display_timestamp(time):
    assert not isinstance(time, np.ndarray), 'Invalid scalar: {}'.format(time)
    scale, units = pick_time_units(time)
    return '{:.2f}{}'.format(time * scale, units)

def pick_time_units(time):
    assert not isinstance(time, np.ndarray), 'Invalid scalar: {}'.format(time)
    if abs(time) < 1:
        return 1000, 'ms'
    else:
        return 1, 's'
