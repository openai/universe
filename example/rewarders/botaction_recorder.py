#!/usr/bin/env python
"""
  This is a small server that accepts connections on a websocket port and writes it to a file.

  The purpose is to allow a universe-env with a built-in bot to record the actions it's taking
  as a demonstration. So the demonstration includes a botactions.jsonl file that gets used instead
  of the vnc client log. (The vnc client log is still recorded and needed to fully parse the VNC
  protocol.)

  It's much simpler than reward_recorder.py, because it doesn't have to also talk to the agent.
  It just takes json messages over a websocket and appends them separated by newlines to the log file.

  The ws port is 15986 unless overridden with --listen-address
  The log file is /tmp/demo/botactions.jsonl unless overridden with --botaction-logfile
"""
import argparse
import logging
import sys
import json
from autobahn.twisted import websocket
from universe.twisty import reactor
logger = logging.getLogger()

class BotactionRecordingServer(websocket.WebSocketServerProtocol, object):

    _next_id = 1
    @classmethod
    def next_id(cls):
        id = cls._next_id
        cls._next_id += 1
        return id

    logfile_path='/tmp/demo/botactions.jsonl'

    def __init__(self):
        super(BotactionRecordingServer, self).__init__()
        self.id = self.next_id()
        self._closed = False
        self.file = None

        logger.info("[BotactionRecordingServer] [%d] Wrote version number", self.id)

    def _emit(self, rec):
        if self.file:
            self.file.write(json.dumps(rec) + '\n');
            self.file.flush()

    def onConnect(self, request):
        logger.info('[BotactionRecordingServer] [%d] Client connecting: %s. Writing to %s', self.id, request.peer, self.logfile_path)
        self.file = open(self.logfile_path, 'w', encoding='utf-8')
        self._emit({
            'version': 1,
            'session_id': self.id,
            '_debug_version': '0.0.1',  # Give this an internal version for debugging corrupt reward.demo files # TODO, pull this from setup.py or the host docker image
        })

    def onOpen(self):
        logger.info("[BotactionRecordingServer] [%d] Websocket connection established", self.id)

    def onClose(self, wasClean, code, reason):
        logger.info('[BotactionRecordingServer] [%d] Client connection closed: %s', self.id, reason)
        if self.file:
            self.file.close()
            self.file = None

        self._closed = True

    def onMessage(self, msg, binary):
        logger.debug('[BotactionRecordingServer] [%d] Received message from client: %s', self.id, msg)

        self._emit(json.loads(msg.decode('utf-8')));

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-l', '--listen-address', default='127.0.0.1:15896', help='Address to listen on')
    parser.add_argument('-o', '--botaction-logfile', default='/tmp/demo/botactions.jsonl', help='Filename for timestamped log of bot actions.')
    args = parser.parse_args()

    BotactionRecordingServer.logfile_path = args.botaction_logfile

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    factory = websocket.WebSocketServerFactory()
    factory.protocol = BotactionRecordingServer

    host, port = args.listen_address.split(':')
    port = int(port)
    logger.info('Listening on %s:%s', host, port)
    reactor.listenTCP(port, factory)
    reactor.run()
    return 0

if __name__ == '__main__':
    sys.exit(main())
