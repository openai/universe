from universe.envs import vnc_env

class FlashgamesEnv(vnc_env.VNCEnv):
     def __init__(self):
        super(FlashgamesEnv, self).__init__()
        self._probe_key = 0x60  # backtick `
