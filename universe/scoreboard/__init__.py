from gym.benchmarks import register_benchmark

register_benchmark(
    id='Atari7VNC-v0',
    score_method='average_last_100_episodes',
    tasks={
        "BeamRider": {
            "env_id": "VNCBeamRider-v3",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Breakout": {
            "env_id": "VNCBreakout-v3",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Enduro": {
            "env_id": "VNCEnduro-v3",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Pong": {
            "env_id": "gym-core.Pong-v3",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Qbert": {
            "env_id": "VNCQbert-v3",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Seaquest": {
            "env_id": "VNCSeaquest-v3",
            "seeds": 1,
            "timesteps": 10000000
        },
        "SpaceInvaders": {
            "env_id": "VNCSpaceInvaders-v3",
            "seeds": 1,
            "timesteps": 10000000
        }
    })
