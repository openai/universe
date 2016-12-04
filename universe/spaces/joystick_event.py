class JoystickEvent(object):
    pass


class JoystickAxisEvent(JoystickEvent):
    def __init__(self, amount):
        self.amount = float(amount)

    def __repr__(self):
        return str(type(self)) + '<amount={}>'.format(self.amount)

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return self.amount.__hash__()

    def __eq__(self, other):
        return type(other) == type(self) and \
               other.amount == self.amount

    def compile(self):
        return type(self).__name__, self.amount


class JoystickAxisXEvent(JoystickAxisEvent):
    pass


class JoystickAxisYEvent(JoystickAxisEvent):
    pass


class JoystickAxisZEvent(JoystickAxisEvent):
    pass


class JoystickAxisRxEvent(JoystickAxisEvent):
    pass


class JoystickAxisRyEvent(JoystickAxisEvent):
    pass


class JoystickAxisRzEvent(JoystickAxisEvent):
    pass


class JoystickSlider0Event(JoystickAxisEvent):
    pass


class JoystickSlider1Event(JoystickAxisEvent):
    pass
