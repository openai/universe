import gym
import universe
from universe import wrappers

def test_joint():
    env1 = gym.make('test.DummyVNCEnv-v0')
    env2 = gym.make('test.DummyVNCEnv-v0')
    env1.configure(_n=3)
    env2.configure(_n=3)
    for reward_buffer in [env1._reward_buffers[0], env2._reward_buffers[0]]:
        reward_buffer.set_env_info('running', 'test.DummyVNCEnv-v0', '1', 60)
        reward_buffer.reset('1')
        reward_buffer.push('1', 10, False, {})

    env = wrappers.Joint([env1, env2])
    env.configure()
    assert env.n == 6
    observation_n = env.reset()
    assert observation_n == [None] * 6

    observation_n, reward_n, done_n, info = env.step([[] for _ in range(env.n)])
    assert reward_n == [10.0, 0.0, 0.0, 10.0, 0.0, 0.0]
    assert done_n == [False] * 6
