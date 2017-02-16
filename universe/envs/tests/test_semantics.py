# import numpy as np
# from universe import vectorized

# class SimpleEnv(vectorized.Env):
#     def _step(self, action_n):
#         return {'vision': np.zeros((10, 10))}, 10, False, {}


# from universe.envs import vnc_env
import gym
import numpy as np
import os
import universe

from autobahn.twisted import websocket
from PIL import Image

from universe.rewarder import rewarder_client, rewarder_session, reward_buffer
from universe import spaces, wrappers

def get_rewarder_session(env):
    return env.unwrapped.rewarder_session

def get_vnc_session(env):
    return env.unwrapped.vnc_session

def get_rewarder_client(env):
    rewarder_session = get_rewarder_session(env)
    return rewarder_session.connections['0'].rewarder_client

def get_reward_buffer(env):
    rewarder_session = get_rewarder_session(env)
    return rewarder_session.connections['0'].reward_buffer

def setup_module(module):
    universe.configure_logging('-')

class FakeVNCConnection(object):
    def __init__(self, name, address, password, encoding=None, fine_quality_level=None, subsample_level=None, start_timeout=None):
        self.name = name
        self.address = address
        self.password = password
        self.encoding = encoding
        self.fine_quality_level = fine_quality_level
        self.subsample_level = subsample_level
        self.start_timeout = start_timeout

        self._frame = np.array(Image.open(os.path.join(os.path.dirname(__file__), 'dusk-drive.png')))

    def step(self, action):
        info_d = {}
        return self._frame, info_d

    def _to_dict(self):
        return {
            'name': self.name,
            'address': self.address,
            'password': self.password,
            'encoding': self.encoding,
            'fine_quality_level': self.fine_quality_level,
            'subsample_level': self.subsample_level,
            'start_timeout': self.start_timeout,
        }

class FakeVNCSession(object):
    def __init__(self):
        self.connections = {}

    def connect(self, name, **kwargs):
        self.connections[name] = FakeVNCConnection(name=name, **kwargs)

    def close(self, name=None):
        if name is not None:
            del self.connections[name]
        else:
            self.connections = None

    def step(self, action_d):
        observation_d = {}
        info_d = {}
        err_d = {}
        for name, conn in self.connections.items():
            observation, info = conn.step(action_d.get(name, []))
            observation_d[name] = observation
            info_d[name] = info
        return observation_d, info_d, err_d

    def _to_dict(self):
        return {name: conn._to_dict() for name, conn in self.connections.items()}

class FakeRewarderConnection(object):
    def __init__(self, name, address, label, password, env_id=None, seed=None, fps=60,
                 start_timeout=None, observer=False, skip_network_calibration=False):
        self.name = name
        self.address = address
        self.label = label
        self.password = password
        self.env_id = env_id
        self.seed = None
        self.fps = fps
        self.start_timeout = start_timeout
        self.observer = observer
        self.skip_network_calibration = skip_network_calibration

        self.reward_buffer = reward_buffer.RewardBuffer(label=self.label)

        factory = websocket.WebSocketClientFactory('ws://'+address)
        factory.reward_buffer = self.reward_buffer
        factory.label = self.label
        self.rewarder_client = rewarder_client.RewarderClient()
        self.rewarder_client.factory = factory
        self.rewarder_client.onConnect(None)

    def reset(self, seed=None):
        self.seed = seed

    def pop(self, peek=False):
        return self.reward_buffer.pop(peek=peek)

    def _to_dict(self):
        return {
            'name': self.name,
            'address': self.address,
            'label': self.label,
            'password': self.password,
            'env_id': self.env_id,
            'seed': self.seed,
            'fps': self.fps,
            'start_timeout': self.start_timeout,
            'observer': self.observer,
            'skip_network_calibration': self.skip_network_calibration,
        }

class FakeRewarder(object):
    def __init__(self):
        self.connections = {}

    def reset(self, seed=None, **kwargs):
        for conn in self.connections.values():
            conn.reset(seed=seed)

    def connect(self, name, **kwargs):
        self.connections[name] = FakeRewarderConnection(name=name, **kwargs)
        return rewarder_session.Network()

    def close(self, name=None):
        if name is not None:
            del self.connections[name]
        else:
            self.connections = None

    def pop(self, peek_d):
        reward_d = {}
        done_d = {}
        info_d = {}
        err_d = {}
        for name, conn in self.connections.items():
            reward, done, info = conn.pop(peek=peek_d.get(name))
            reward_d[name] = reward
            done_d[name] = done
            info_d[name] = info
        return reward_d, done_d, info_d, err_d

    def _to_dict(self):
        return {name: conn._to_dict() for name, conn in self.connections.items()}

def test_connect():
    env = gym.make('flashgames.DuskDrive-v0')
    env.configure(vnc_driver=FakeVNCSession, rewarder_driver=FakeRewarder, remotes='vnc://example.com:5900+15900')
    vnc_session = get_vnc_session(env)
    rewarder_session = get_rewarder_session(env)

    assert vnc_session._to_dict() == {'0': {'name': '0', 'subsample_level': 2, 'encoding': 'tight', 'fine_quality_level': 50, 'start_timeout': 7, 'address': 'example.com:5900', 'password': 'openai'}}
    assert rewarder_session._to_dict() == {'0': {'start_timeout': 7, 'seed': None, 'name': '0', 'fps': 60, 'address': 'example.com:15900', 'env_id': 'flashgames.DuskDrive-v0', 'password': 'openai', 'skip_network_calibration': False, 'observer': False, 'label': '0:example.com:5900'}}

def test_describe_handling():
    env = gym.make('flashgames.DuskDrive-v0')
    env.configure(vnc_driver=FakeVNCSession, rewarder_driver=FakeRewarder, remotes='vnc://example.com:5900+15900')
    env.reset()

    reward_buffer = get_reward_buffer(env)
    rewarder_client = get_rewarder_client(env)

    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'resetting', 'fps': 60}, {'episode_id': '1'})

    assert reward_buffer._remote_episode_id == '1'
    assert reward_buffer._remote_env_state == 'resetting'
    assert reward_buffer._current_episode_id == None
    assert reward_buffer.reward_state(reward_buffer._current_episode_id)._env_state == None

    rewarder_client._manual_recv('v0.reply.env.reset', {}, {'episode_id': '1'})

    assert reward_buffer._remote_episode_id == '1'
    assert reward_buffer._remote_env_state == 'resetting'
    assert reward_buffer._current_episode_id == '1'
    assert reward_buffer.reward_state(reward_buffer._current_episode_id)._env_state == 'resetting'

def test_vnc_env():
    env = gym.make('flashgames.DuskDrive-v0')
    env = wrappers.Unvectorize(env)
    env.configure(vnc_driver=FakeVNCSession, rewarder_driver=FakeRewarder, remotes='vnc://example.com:5900+15900')
    env.reset()

    rewarder_client = get_rewarder_client(env)

    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'resetting', 'fps': 60}, {'episode_id': '1'})

    observation, reward, done, info = env.step([spaces.KeyEvent.by_name('a', down=True)])
    assert (observation, reward, done, info['env_status.env_state'], info['env_status.episode_id']) == (None, 0, False, None, None)

    rewarder_client._manual_recv('v0.reply.env.reset', {}, {'episode_id': '1'})

    observation, reward, done, info = env.step([spaces.KeyEvent.by_name('a', down=True)])
    assert (observation, reward, done, info['env_status.env_state'], info['env_status.episode_id']) == (None, 0, False, 'resetting', '1')

    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'running', 'fps': 60}, {'episode_id': '1'})

    rewarder_client._manual_recv('v0.env.reward', {'reward': 10, 'done': False, 'info': {}}, {'episode_id': '1'})
    rewarder_client._manual_recv('v0.env.reward', {'reward': 15, 'done': False, 'info': {}}, {'episode_id': '1'})
    rewarder_client._manual_recv('v0.env.reward', {'reward': -3, 'done': False, 'info': {}}, {'episode_id': '1'})

    observation, reward, done, info = env.step([spaces.KeyEvent.by_name('a', down=True)])
    assert sorted(observation.keys()) == ['text', 'vision']
    assert observation['text'] == []
    assert observation['vision'].shape == (768, 1024, 3)
    assert (reward, done, info['env_status.env_state'], info['env_status.episode_id']) == (22, False, 'running', '1')
    assert info['stats.reward.count'] == 3

def test_boundary_simple():
    env = gym.make('flashgames.DuskDrive-v0')
    env = wrappers.Unvectorize(env)
    env.configure(vnc_driver=FakeVNCSession, rewarder_driver=FakeRewarder, remotes='vnc://example.com:5900+15900')
    env.reset()

    rewarder_client = get_rewarder_client(env)
    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'resetting', 'fps': 60}, {'episode_id': '1'})
    rewarder_client._manual_recv('v0.reply.env.reset', {}, {'episode_id': '1'})

    rewarder_client._manual_recv('v0.env.reward', {'reward': 1, 'done': False, 'info': {}}, {'episode_id': '1'})
    rewarder_client._manual_recv('v0.env.reward', {'reward': 2, 'done': True, 'info': {}}, {'episode_id': '1'})
    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'resetting', 'fps': 60}, {'episode_id': '2'})

    # We have reward of 3 for episode 1, and episode 2 should now be resetting
    observation, reward, done, info = env.step([])
    assert info['mask.masked.observation']
    assert info['mask.masked.action']
    assert (reward, done, info['env_status.env_state'], info['env_status.episode_id']) == (3, True, 'resetting', '2')

def test_boundary_multiple():
    env = gym.make('flashgames.DuskDrive-v0')
    env = wrappers.Unvectorize(env)
    env.configure(vnc_driver=FakeVNCSession, rewarder_driver=FakeRewarder, remotes='vnc://example.com:5900+15900')
    env.reset()

    rewarder_client = get_rewarder_client(env)
    # episode 2
    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'resetting', 'fps': 60}, {'episode_id': '2'})
    rewarder_client._manual_recv('v0.reply.env.reset', {}, {'episode_id': '2'})
    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'running', 'fps': 60}, {'episode_id': '2'})
    rewarder_client._manual_recv('v0.env.reward', {'reward': 2, 'done': True, 'info': {}}, {'episode_id': '2'})

    # episode 3
    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'resetting', 'fps': 60}, {'episode_id': '3'})
    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'running', 'fps': 60}, {'episode_id': '3'})
    rewarder_client._manual_recv('v0.env.reward', {'reward': 3, 'done': True, 'info': {}}, {'episode_id': '3'})

    # episode 4
    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'resetting', 'fps': 60}, {'episode_id': '4'})
    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'running', 'fps': 60}, {'episode_id': '4'})
    rewarder_client._manual_recv('v0.env.reward', {'reward': 4, 'done': False, 'info': {}}, {'episode_id': '4'})

    observation, reward, done, info = env.step([])
    assert not info.get('mask.masked.observation')
    assert not info.get('mask.masked.action')
    assert (reward, done, info['env_status.env_state'], info['env_status.episode_id']) == (2, True, 'running', '4')
    assert (info['env_status.complete.env_state'], info['env_status.complete.episode_id']) == ('running', '2')

    observation, reward, done, info = env.step([])
    assert (reward, done, info['env_status.env_state'], info['env_status.episode_id']) == (4, False, 'running', '4')

def test_peek():
    env = gym.make('flashgames.DuskDrive-v0')
    env = wrappers.Unvectorize(env)
    env.configure(vnc_driver=FakeVNCSession, rewarder_driver=FakeRewarder, remotes='vnc://example.com:5900+15900')
    env.reset()

    rewarder_client = get_rewarder_client(env)
    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'resetting', 'fps': 60}, {'episode_id': '1'})
    rewarder_client._manual_recv('v0.reply.env.reset', {}, {'episode_id': '1'})

    observation, reward, done, info = env.step([spaces.PeekReward])

    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'resetting', 'fps': 60}, {'episode_id': '2'})
    observation, reward, done, info = env.step([spaces.PeekReward])
    assert info['mask.masked.observation']
    assert info['mask.masked.action']
    assert info['env_status.episode_id'] == '1'
    assert info['env_status.env_state'] == 'resetting'
    assert info['env_status.peek.episode_id'] == '2'
    assert info['env_status.peek.env_state'] == 'resetting'

    rewarder_client._manual_recv('v0.env.describe', {'env_id': 'flashgames.DuskDrive-v0', 'env_state': 'running', 'fps': 60}, {'episode_id': '2'})
    observation, reward, done, info = env.step([spaces.PeekReward])
    assert not info.get('mask.masked.observation')
    assert not info.get('mask.masked.action')
    assert info['env_status.episode_id'] == '1'
    assert info['env_status.env_state'] == 'resetting'
    assert info['env_status.peek.episode_id'] == '2'
    assert info['env_status.peek.env_state'] == 'running'
