import threading
from twisted.python.runtime import platform

# On OSX, we should use kqueue rather than the default select
# backend. (Proximal issue is that select only can handle a limited
# number of file descriptors.)
#
# Based off twisted.internet.default
def _get_reactor(platform):
    try:
        if platform.isLinux():
            try:
                from twisted.internet import epollreactor
                cls = epollreactor.EPollReactor
            except ImportError:
                from twisted.internet import pollreactor
                cls = pollreactor.PollReactor
        elif platform.isMacOSX():
            from twisted.internet import kqreactor
            cls = kqreactor.KQueueReactor
        elif platform.getType() == 'posix' and not platform.isMacOSX():
            from twisted.internet import pollreactor
            cls = pollreactor.PollReactor
        else:
            from twisted.internet import selectreactor
            cls = selectreactor.SelectReactor
    except ImportError:
        from twisted.internet import selectreactor
        cls = selectreactor.SelectReactor
    return cls()

class TwistedThread(threading.Thread):
    started = False
    daemon = True

    @classmethod
    def start_once(cls):
        if cls.started:
            return
        cls.started = True

        instance = cls(name='Twisted')
        instance.start()

    def run(self):
        reactor.run(installSignalHandlers=False)

reactor = _get_reactor(platform)
start_once = TwistedThread.start_once
