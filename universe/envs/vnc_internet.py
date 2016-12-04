from universe.envs import vnc_env

class InternetEnv(vnc_env.VNCEnv):
     def __init__(self):
        super(InternetEnv, self).__init__()
        self._probe_key = 0x60  # backtick `
