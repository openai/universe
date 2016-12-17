from gym.benchmarks import scoring
from gym.benchmarks import register_benchmark

register_benchmark(
    id='Atari7VNC-v0',
    scorer=scoring.TotalReward(),
    name='AtariVNC',
    description='7 Atari games, with pixel observations (using universe)',
    tasks=[
        {
            "env_id": "VNCBeamRider-v3",
            "trials": 1,
            "max_timesteps": 10000000
        },
        {
            "env_id": "VNCBreakout-v3",
            "trials": 1,
            "max_timesteps": 10000000
        },
        {
            "env_id": "VNCEnduro-v3",
            "trials": 1,
            "max_timesteps": 10000000
        },
        {
            "env_id": "gym-core.Pong-v3",
            "trials": 1,
            "max_timesteps": 10000000
        },
        {
            "env_id": "VNCQbert-v3",
            "trials": 1,
            "max_timesteps": 10000000
        },
        {
            "env_id": "VNCSeaquest-v3",
            "trials": 1,
            "max_timesteps": 10000000
        },
        {
            "env_id": "VNCSpaceInvaders-v3",
            "trials": 1,
            "max_timesteps": 10000000
        }
    ])

register_benchmark(
    id='FlashRacing-v0',
    scorer=scoring.RewardPerTime(),
    name='FlashRacing',
    description='7 flash racing games, goal is best score per time',
    tasks=[
        {'env_id': 'flashgames.NeonRace-v0',
         'trials': 1,
         'max_timesteps': 5000000,
         'reward_floor':   175.0,
         'reward_ceiling': 1700.0,
        },
        {'env_id': 'flashgames.CoasterRacer-v0',
         'trials': 1,
         'max_timesteps': 5000000,
         'reward_floor':   17.0,
         'reward_ceiling': 400.0,
        },
        {'env_id': 'flashgames.HeatRushUsa-v0',
         'trials': 1,
         'max_timesteps': 5000000,
         'reward_floor':   150.0,
         'reward_ceiling': 700.0,
        },
        {'env_id': 'flashgames.FormulaRacer-v0',
         'trials': 1,
         'max_timesteps': 5000000,
         'reward_floor':  0.27,
         'reward_ceiling': 1.0,
        },
        {'env_id': 'flashgames.DuskDrive-v0',
         'trials': 1,
         'max_timesteps': 5000000,
         'reward_floor':   5000.0,
         'reward_ceiling': 15000.0,
        },
        {'env_id': 'flashgames.SpacePunkRacer-v0',
         'trials': 1,
         'max_timesteps': 5000000,
         'reward_floor':   0.67,
         'reward_ceiling': 2.25,
        },
        {'env_id': 'flashgames.NeonRace2-v0',
         'trials': 1,
         'max_timesteps': 5000000,
         'reward_floor':   0.0,
         'reward_ceiling': 1200.0,
        }
    ])
