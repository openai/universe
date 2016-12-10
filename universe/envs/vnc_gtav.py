from universe.envs import vnc_env
from universe.spaces.joystick_action_space import JoystickActionSpace


class GTAVEnv(vnc_env.VNCEnv):
    def __init__(self):
        super(GTAVEnv, self).__init__()
        self.action_space = JoystickActionSpace(axis_x=True, axis_z=True)
        self._send_actions_over_websockets = True
        self._skip_network_calibration = True

