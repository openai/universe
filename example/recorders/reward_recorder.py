#!/usr/bin/env python
import argparse
import logging
import sys

from autobahn.twisted import websocket
from universe.rewarder import reward_proxy_server
from universe.twisty import reactor

logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-l', '--listen-address', default='0.0.0.0:15898', help='Address to listen on')
    parser.add_argument('-s', '--rewarder-address', default='127.0.0.1:15900', help='Address of the reward server to run on.')
    parser.add_argument('-d', '--logfile-dir', default=None, help='Base directory to write logs for each connection')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    factory = websocket.WebSocketServerFactory()
    factory.protocol = reward_proxy_server.RewardProxyServer
    factory.rewarder_address = args.rewarder_address
    factory.logfile_dir = args.logfile_dir
    factory.setProtocolOptions(maxConnections=1)  # We only write reward logs to one place, so only allow one connection

    host, port = args.listen_address.split(':')
    port = int(port)
    logger.info('Listening on %s:%s', host, port)
    reactor.listenTCP(port, factory)
    reactor.run()
    return 0

if __name__ == '__main__':
    sys.exit(main())
