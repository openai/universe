from universe import error
import six

def merge_infos(info1, info2):
    """We often need to aggregate together multiple infos. Most keys can
    just be clobbered by the new info, but e.g. any keys which contain
    counts should be added. The merge schema is indicated by the key
    namespace.

    Namespaces:

    - stats.timers: Timing
    - stats.gauges: Gauge values
    - stats.*: Counts of a quantity
    """
    for key, value in six.iteritems(info2):
        if key in info1 and key.startswith('stats'):
            if key.startswith('stats.timers'):
                # timer
                info1[key] += value
            elif key.startswith('stats.gauges'):
                # gauge
                info1[key] = value
            else:
                # counter
                info1[key] += value
        else:
            info1[key] = value

def merge_reward_n(accum_reward_n, reward_n):
    for i in range(len(reward_n)):
        if reward_n[i] is not None:
            # Add rewards
            accum_reward_n[i] += reward_n[i]

def merge_done_n(accum_done_n, done_n):
    for i in range(len(done_n)):
        # Copy over done if the episode is indeed none
        if done_n[i]:
            accum_done_n[i] = done_n[i]

def _merge_observation(accum_observation, observation):
    """
    Old visual observation is discarded, because it is outdated frame.
    Text observations are merged, because they are messages sent from the rewarder.
    """
    if observation is None:
        # We're currently masking. So accum_observation probably
        # belongs to the previous episode. We may lose a "text"
        # observation from the previous episode, but that's ok.
        return None
    elif accum_observation is None:
        # Nothing to merge together
        return observation

    accum_observation['vision'] = observation.get('vision')
    accum_observation['text'] = accum_observation.get('text', []) + observation.get('text', [])
    return accum_observation

def merge_observation_n(accum_observation_n, observation_n):
    # Merge observations.
    for i in range(len(accum_observation_n)):
        accum_observation_n[i] = _merge_observation(accum_observation_n[i], observation_n[i])

def merge_n(
        accum_observation_n, accum_reward_n, accum_done_n, accum_info,
        observation_n, reward_n, done_n, info,
):
    # Merge observation/reward/done
    merge_observation_n(accum_observation_n, observation_n)
    merge_reward_n(accum_reward_n, reward_n)
    merge_done_n(accum_done_n, done_n)

    # Merge together infos. We deep merge the 'n' key and do a
    # simple merge on everything else.
    accum_info_n = accum_info['n']
    for accum_info_i, info_i in zip(accum_info_n, info['n']):
        merge_infos(accum_info_i, info_i)

    merge_infos(accum_info, info)
    accum_info['n'] = accum_info_n
