# a proxy server that handles both reward channel and vnc.
from twisted.python import log
from autobahn.twisted import websocket
import logging
import os
import time
import pexpect
import sys
import threading

from universe.vncdriver.vnc_proxy_server import VNCProxyServer
from universe.rewarder.reward_proxy_server import RewardProxyServer
from universe import utils

logger = logging.getLogger(__name__)


class DualProxyServer(VNCProxyServer):
    def __init__(self, action_queue=None, error_buffer=None, enable_logging=True):
        self._log_info('DualProxyServer inited')
        self.reward_proxy = None

        super(DualProxyServer, self).__init__(action_queue, error_buffer, enable_logging)

    def _log_info(self, msg, *args, **kwargs):
        logger.info('[dual_proxy] ' + msg, *args, **kwargs)

    def recv_ClientInit(self, block):
        # start reward proxy.
        self._log_info('Starting reward proxy server')
        self.reward_proxy = pexpect.spawnu(self.factory.reward_proxy_bin,
                                           logfile=sys.stdout,
                                           timeout=None)

        # wait on reward proxy to be up.
        self._log_info('Waiting for reward proxy server')
        self.reward_proxy.expect('\[RewardProxyServer\]')
        self.reward_proxy_thread = threading.Thread(target=lambda: self.reward_proxy.expect(pexpect.EOF))
        self.reward_proxy_thread.start()

        self._log_info('Reward proxy server is up %s', self.reward_proxy.before)

        super(DualProxyServer, self).recv_ClientInit(block)

        self.logfile_dir = self.log_manager.logfile_dir

    def close(self):
        # end connections.
        super(DualProxyServer, self).close()

        # wait for rewarder to close.
        if self.reward_proxy:
            self.reward_proxy.terminate()

        # upload to s3.
        # probably hacky right now.
        logger.info('log manager = %s', self.log_manager)
        if self.log_manager:
            os.system('/app/universe/bin/upload_directory.sh demonstrator_%(recorder_id)s %(directory)s %(bucket)s' %
                    dict(recorder_id=self.factory.recorder_id, directory=self.logfile_dir,
                        bucket=self.factory.bucket)
                    )

