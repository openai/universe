from universe import wrappers
import gym
import universe
from universe.utils import ErrorBuffer

env = gym.make('wob.mini.TicTacToe-v0')

env.configure(remotes='vnc://localhost:5900+15900')
env.reset(query_id='non-parametric bayes models are gross')

while True:
    observation_n, reward_n, done_n, info = env.step([[]])
