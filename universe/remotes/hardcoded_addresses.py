import logging
import os
import re
import six.moves.urllib.parse as urlparse

from universe import error, utils
from universe.remotes import remote

logger = logging.getLogger(__name__)

class HardcodedAddresses(object):
    @classmethod
    def build(cls, remotes, **kwargs):
        parsed = urlparse.urlparse(remotes)
        if parsed.scheme != 'vnc':
            raise error.Error('HardcodedAddresses must be initialized with a string starting with vnc://: {}'.format(remotes))

        addresses = parsed.netloc.split(',')
        query = urlparse.parse_qs(parsed.query)
        # We could support per-backend passwords, but no need for it
        # right now.
        password = query.get('password', [utils.default_password()])[0]
        vnc_addresses, rewarder_addresses = parse_remotes(addresses)
        res = cls(vnc_addresses, rewarder_addresses, vnc_password=password, rewarder_password=password, **kwargs)
        return res, res.available_n

    def __init__(self, vnc_addresses, rewarder_addresses, vnc_password, rewarder_password, start_timeout=None):
        if vnc_addresses is not None:
            self.available_n = len(vnc_addresses)
        elif rewarder_addresses is not None:
            self.available_n = len(rewarder_addresses)
        else:
            assert False

        self.supports_reconnect = False
        self.connect_vnc = vnc_addresses is not None
        self.connect_rewarder = rewarder_addresses is not None
        if rewarder_addresses is None:
            logger.info("No rewarder addresses were provided, so this env cannot connect to the remote's rewarder channel, and cannot send control messages (e.g. reset)")

        self.vnc_addresses = vnc_addresses
        self.vnc_password = vnc_password
        self.rewarder_addresses = rewarder_addresses
        self.rewarder_password = rewarder_password
        if start_timeout is None:
            start_timeout = 2 * self.available_n + 5
        self.start_timeout = start_timeout

        self._popped = False

    def pop(self, n=None):
        if self._popped:
            assert n is None
            return []
        self._popped = True

        remotes = []
        for i in range(self.available_n):
            if self.vnc_addresses is not None:
                vnc_address = self.vnc_addresses[i]
            else:
                vnc_address = None

            if self.rewarder_addresses is not None:
                rewarder_address = self.rewarder_addresses[i]
            else:
                rewarder_address = None

            name = self._handles[i]
            env = remote.Remote(
                handle=self._handles[i],
                vnc_address=vnc_address,
                vnc_password=self.vnc_password,
                rewarder_address=rewarder_address,
                rewarder_password=self.rewarder_password,
            )
            remotes.append(env)
        return remotes

    def allocate(self, handles, initial=False, params={}):
        if len(handles) > self.available_n:
            raise error.Error('Requested {} handles, but only have {} envs'.format(len(handles), self.available_n))
        self.n = len(handles)
        self._handles = handles

    def close(self):
        pass

def parse_remotes(remotes):
    # Parse a list of remotes of the form:
    #
    # address:vnc_port+rewarder_port (e.g. localhost:5900+15900)
    #
    # either vnc_port or rewarder_port can be omitted, but not both

    all_vnc = None
    all_rewarder = None

    vnc_addresses = []
    rewarder_addresses = []

    for remote in remotes:
        # Parse off +, then :
        if '+' in remote:
            if all_vnc == False:
                raise error.Error('Either all or no remotes must have rewarders: {}'.format(remotes))
            all_vnc = True

            remote, rewarder_port = remote.split('+')
            if not re.match(r'^[0-9]+$', rewarder_port):
                raise error.Error('Rewarder port must be an integer, not `{}`: {}'.format(rewarder_port, remotes))
            rewarder_port = int(rewarder_port)
        else:
            if all_vnc == True:
                raise error.Error('Either all or no remotes must have rewarders: {}'.format(remotes))
            all_vnc = False

            rewarder_port = None

        if ':' in remote:
            if all_rewarder == False:
                raise error.Error('Either all or no remotes must have a VNC port: {}'.format(remotes))
            all_rewarder = True

            remote, vnc_port = remote.split(':')
            if not re.match(r'^[0-9]+$', vnc_port):
                raise error.Error('VNC port must be an integer, not `{}`: {}'.format(vnc_port, remotes))
            vnc_port = int(vnc_port)
        else:
            if all_rewarder == True:
                raise error.Error('Either all or no remotes must have a VNC port: {}'.format(remotes))
            all_rewarder = False

            vnc_port = None
            all_rewarder = False

        host = remote
        if not re.match(r'^[-a-zA-Z0-9\.\_]+$', host):
            raise error.Error('Invalid hostname for remote: {}'.format(remotes))

        if rewarder_port is not None:
            rewarder_address = '{}:{}'.format(host, rewarder_port)
            rewarder_addresses.append(rewarder_address)

        if vnc_port is not None:
            vnc_address = '{}:{}'.format(host, vnc_port)
            vnc_addresses.append(vnc_address)

    if not all_vnc and not all_rewarder:
        raise error.Error('You must provide either rewarder or a VNC port: {}'.format(remotes))

    if not vnc_addresses:
        vnc_addresses = None
    if not rewarder_addresses:
        rewarder_addresses = None
    return vnc_addresses, rewarder_addresses
