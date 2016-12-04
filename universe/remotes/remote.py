class Remote(object):
    def __init__(self, handle, vnc_address, vnc_password, rewarder_address, rewarder_password, name=None):
        self.name = name
        self.handle = handle
        self.vnc_address = vnc_address
        self.vnc_password = vnc_password
        self.rewarder_address = rewarder_address
        self.rewarder_password = rewarder_password

    def __str__(self):
        return 'Remote<{}:{}>'.format(self.handle, self.name)

    def __repr__(self):
        return str(self)
