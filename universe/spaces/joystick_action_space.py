import gym
from gym.spaces import Box
from universe.spaces import joystick_event
from gym.spaces import prng
from collections import OrderedDict


class JoystickActionSpace(gym.Space):
    """
    Programmable joystick - currently Windows-only => mapped to vJoy
    """
    def __init__(self, axis_x=False, axis_y=False, axis_z=False, axis_rx=False, axis_ry=False, axis_rz=False,
                 slider_0=False, slider_1=False):
        self.event_space_map = OrderedDict()

        if axis_x:
            self.axis_x = box_axis()
            self.event_space_map[joystick_event.JoystickAxisXEvent] = self.axis_x
        if axis_y:
            self.axis_y = box_axis()
            self.event_space_map[joystick_event.JoystickAxisYEvent] = self.axis_y
        if axis_z:
            self.axis_z = box_axis()
            self.event_space_map[joystick_event.JoystickAxisZEvent] = self.axis_z
        if axis_rx:
            self.axis_rx = box_axis()
            self.event_space_map[joystick_event.JoystickAxisRxEvent] = self.axis_rx
        if axis_ry:
            self.axis_ry = box_axis()
            self.event_space_map[joystick_event.JoystickAxisRyEvent] = self.axis_ry
        if axis_rz:
            self.axis_rz = box_axis()
            self.event_space_map[joystick_event.JoystickAxisRzEvent] = self.axis_rz
        if slider_0:
            self.slider_0 = box_axis()
            self.event_space_map[joystick_event.JoystickSlider0Event] = self.slider_0
        if slider_1:
            self.slider_1 = box_axis()
            self.event_space_map[joystick_event.JoystickSlider1Event] = self.slider_1
        # TODO: Add buttons (similar to a vnc_event.KeyEvent - but 1..32)
        # TODO: Add POV hats

    def contains(self, action):
        if not isinstance(action, list):
            return False
        for a in action:
            if isinstance(a, joystick_event.JoystickAxisEvent):
                axis = self.event_space_map[a]
                if not axis.contains(a):
                    return False
        return True

    def sample(self):
        event_type_index = prng.np_random.randint(len(self.event_space_map))
        event_type = list(self.event_space_map.keys())[event_type_index]
        if event_type.__bases__[0] == joystick_event.JoystickAxisEvent:
            event = [event_type(self.event_space_map[event_type].sample()[0])]
        else:
            raise JoystickActionSpaceException('Unexpected event type')
        return event


class JoystickActionSpaceException(Exception):
    pass


def box_axis():
    return Box(-1.0, 1.0, shape=(1,))
