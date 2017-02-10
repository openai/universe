import gym
import universe.wrappers.experimental
from universe import envs, spaces
from universe.wrappers import gym_core_sync
from universe.wrappers.blocking_reset import BlockingReset
from universe.wrappers.diagnostics import Diagnostics
from universe.wrappers.gym_core import GymCoreAction, GymCoreObservation, CropAtari
from universe.wrappers.joint import Joint
from universe.wrappers.logger import Logger
from universe.wrappers.monitoring import Monitor
from universe.wrappers.multiprocessing_env import WrappedMultiprocessingEnv, EpisodeID
from universe.wrappers.recording import Recording
from universe.wrappers.render import Render
from universe.wrappers.throttle import Throttle
from universe.wrappers.time_limit import TimeLimit
from universe.wrappers.timer import Timer
from universe.wrappers.vectorize import Vectorize, Unvectorize, WeakUnvectorize
from universe.wrappers.vision import Vision


def wrap(env):
    return Timer(Render(Throttle(env)))

def WrappedVNCEnv():
    return wrap(envs.VNCEnv())

def WrappedGymCoreEnv(gym_core_id, fps=None, rewarder_observation=False):
    # Don't need to store the ID on the instance; it'll be retrieved
    # directly from the spec
    env = wrap(envs.VNCEnv(fps=fps))
    if rewarder_observation:
        env = GymCoreObservation(env, gym_core_id=gym_core_id)
    return env

def WrappedGymCoreSyncEnv(gym_core_id, fps=60, rewarder_observation=False):
    spec = gym.spec(gym_core_id)
    env = gym_core_sync.GymCoreSync(BlockingReset(wrap(envs.VNCEnv(fps=fps))))
    if rewarder_observation:
        env = GymCoreObservation(env, gym_core_id=gym_core_id)
    elif spec._entry_point.startswith('gym.envs.atari:'):
        env = CropAtari(env)

    return env

def WrappedFlashgamesEnv():
    keysym = spaces.KeyEvent.by_name('`').key
    return wrap(envs.VNCEnv(probe_key=keysym))

def WrappedInternetEnv(*args, **kwargs):
    return wrap(envs.InternetEnv(*args, **kwargs))

def WrappedStarCraftEnv(*args, **kwargs):
    return wrap(envs.StarCraftEnv(*args, **kwargs))

def WrappedGTAVEnv(*args, **kwargs):
    return wrap(envs.GTAVEnv(*args, **kwargs))

def WrappedWorldOfGooEnv(*args, **kwargs):
    return wrap(envs.WorldOfGooEnv(*args, **kwargs))
