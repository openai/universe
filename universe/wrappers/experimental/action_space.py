import logging

from universe.wrappers.action_space import SafeActionSpace as _SafeActionSpace

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SafeActionSpace(_SafeActionSpace):
    """
    Recall that every universe environment receives a list of VNC events as action.
    There exist many environments for which the set of relevant action is much smaller
    and is known.   For example, Atari environments have a modest number of keys,
    so this wrapper, when applied to an Atari environment will reduce its action space.
    Doing so is very convenient for research, since today's RL algorithms rely on random
    exploration, which is hurt by small action spaces.  As our algorithms get better
    and we switch to using the raw VNC commands, this wrapper will become less important.

    NOTE: This will be the new location for SafeActionSpace, however the logic must currently remain in
    wrappers.SafeActionSpace in order to maintain backwards compatibility.
    """

    def _deprecation_warning(self):
        # No deprecation warning here because we are using the correct import
        pass
