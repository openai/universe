import sys

class Error(Exception):
    pass

class RPCError(Error):
    pass

class ConnectionError(Error):
    pass

class TimeoutError(Error):
    pass

class AuthenticationError(Error):
    pass
