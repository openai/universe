#!/usr/bin/env python
import argparse
import logging
import os
import re
import sys

from universe import utils
from universe.vncdriver import vnc_proxy_server
from twisted.internet import protocol, reactor

logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-l', '--listen-address', default='0.0.0.0:5899', help='Address to listen on')
    parser.add_argument('-s', '--vnc-address', default='127.0.0.1:5900', help='Address of the VNC server to run on.')
    parser.add_argument('-d', '--logfile-dir', default=None, help='Base directory to write logs for each connection')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    factory = protocol.ServerFactory()
    factory.protocol = vnc_proxy_server.VNCProxyServer
    factory.vnc_address = 'tcp:{}'.format(args.vnc_address)
    factory.logfile_dir = args.logfile_dir
    factory.recorder_id = utils.random_alphanumeric().lower()

    host, port = args.listen_address.split(':')
    port = int(port)

    logger.info('Listening on %s:%s', host, port)
    reactor.listenTCP(port, factory, interface=host)
    reactor.run()
    return 0

if __name__ == '__main__':
    sys.exit(main())
