import glob
import os

import gym.monitoring
from gym.monitoring.tests import helpers
from universe import wrappers

def test_multiprocessing_env_monitoring():
    with helpers.tempdir() as temp:
        env = wrappers.WrappedMultiprocessingEnv('Pong-v3')
        env.configure(n=2)
        env.monitor.start(temp)
        env.reset()
        for i in range(2):
            env.step([0, 0])
        env.close()
        manifests = glob.glob(os.path.join(temp, '*.video.*'))
        assert len(manifests) == 2, 'There are {} manifests: {}'.format(len(manifests), manifests)

        results = gym.monitoring.load_results(temp)
        assert results['env_info']['env_id'] == 'Pong-v3'

def test_vnc_monitoring():
    with helpers.tempdir() as temp:
        env = gym.make('gym-core.Pong-v3')
        env = wrappers.GymCoreAction(env)
        env.configure(remotes=2)
        env.monitor.start(temp, seed_n=[1, 2])
        env.reset()
        for i in range(2):
            env.step([0, 0])
        env.monitor.close()
        env.close()

        results = gym.monitoring.load_results(temp)
        assert results['env_info']['env_id'] == 'gym-core.Pong-v3'
