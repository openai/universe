from universe.envs import vnc_env
from universe.spaces import VNCActionSpace


class WorldOfGooEnv(vnc_env.VNCEnv):
    def __init__(self):
        super(WorldOfGooEnv, self).__init__()
        # TODO: set action space screen shape to match
        # HACK: empty keys list fails for some weird reason, give it an 'a'
        self.action_space = VNCActionSpace(keys=['a'], buttonmasks=[1])
