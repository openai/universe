import logging
import os
import re
import shutil
import struct
import time
import traceback

from universe import pyprofile
from universe.vncdriver import auth, constants, fbs_writer
from twisted.internet import defer, endpoints, protocol, reactor

logger = logging.getLogger(__name__)

class LogManager(object):
    def __init__(self, base_logfile_dir, recorder_id, id):
        self.base_logfile_dir = base_logfile_dir
        self.recorder_id = recorder_id
        self.id = id

        self.logfile_dir = os.path.join(self.base_logfile_dir, '{}-{}-{}'.format(int(time.time()), self.recorder_id, str(self.id)))

        logger.info('[vnc_proxy] logfile_dir = %s', self.logfile_dir)
        # Make the logfile directory that will be written to
        if not os.path.isdir(self.logfile_dir):
            logger.info('[vnc_proxy] [%s] Creating log directory %s', self.id, self.logfile_dir)
            os.makedirs(self.logfile_dir)

    @property
    def global_rewards_logfile(self):
        """
        We assume that there is only a single rewards file that is ever written to.
        This is enforced by the reward_recorder that limits connections to one
        """
        return '/tmp/demo/rewards.demo'

    @property
    def global_botaction_logfile(self):
        """
        We assume that there is only a single botactions file that is ever written to.
        This is enforced by the botaction_recorder that appends everything to one file
        """
        return '/tmp/demo/botactions.jsonl'


    @property
    def global_env_id_file(self):
        """
        We assume that there is only a single rewards file that is ever written to.
        This is enforced by the reward_recorder that limits connections to one
        """
        return '/tmp/demo/env_id.txt'

    @property
    def server_logfile(self):
        return os.path.join(self.logfile_dir, 'server.fbs')

    @property
    def client_logfile(self):
        return os.path.join(self.logfile_dir, 'client.fbs')

    def close(self):
        logger.info("[%s] Copying rewards.demo into this connection's log dir", self.id)

        if os.path.exists(self.global_rewards_logfile):
            # Each recording includes all the rewards that we've seen. We count on the player to
            # crop them to only contain the ones that occurred during this recording

            # TODO: Upload to S3 immediately upon disconnect
            shutil.copyfile(self.global_rewards_logfile, os.path.join(self.logfile_dir, 'rewards.demo'))
        else:
            logger.info("%s does not exist; not copying into recording directory", self.global_rewards_logfile)

        if os.path.exists(self.global_botaction_logfile):
            shutil.copyfile(self.global_botaction_logfile, os.path.join(self.logfile_dir, 'botactions.jsonl'))
        else:
            logger.info("%s does not exist; not copying into recording directory", self.global_botaction_logfile)


        if os.path.exists(self.global_env_id_file):
            shutil.copyfile(self.global_env_id_file, os.path.join(self.logfile_dir, 'env_id.txt'))
        else:
            logger.info("%s does not exist; not copying into recording directory", self.global_env_id_file)

        if os.environ.get('COMPLETED_DEMONSTRATION_DIR'):
            dest = os.path.join(os.environ['COMPLETED_DEMONSTRATION_DIR'], os.path.basename(self.logfile_dir))
            logger.info('copying to %s', dest)
            shutil.copytree(self.logfile_dir, dest)
            shutil.copystat(self.logfile_dir, dest)


class VNCProxyServer(protocol.Protocol, object):
    """Bytes received from the end user. (So received data are mostly
    actions.)"""

    _next_id = 0

    SUPPORTED_ENCODINGS = {
        # Maybe we can do copy-rect at some point. May not help with
        # much though.
        # constants.COPY_RECTANGLE_ENCODING,
        constants.TIGHT_ENCODING,
        constants.RAW_ENCODING,
        constants.ZLIB_ENCODING,
        # constants.HEXTILE_ENCODING,
        # constants.CORRE_ENCODING,
        # constants.RRE_ENCODING,
        constants.PSEUDO_CURSOR_ENCODING
    }

    if os.getenv('VNC_ENCODINGS'):
        for enc in os.getenv('VNC_ENCODINGS').split():
            SUPPORTED_ENCODINGS.add(getattr(constants, enc.upper() + '_ENCODING'))
    add_pseudo_cursor_encoding = True
    if os.getenv('VNC_ENCODINGS_NO_ADD_PSEUDO_CURSOR_ENCODING'):
        add_pseudo_cursor_encoding = False

    # ZRLE format is much smaller, but slower to read.
    # SUPPORTED_ENCODINGS.add(constants.ZRLE_ENCODING)

    @classmethod
    def next_id(cls):
        id = cls._next_id
        cls._next_id += 1
        return id

    def __init__(self, action_queue=None, error_buffer=None, enable_logging=True):
        self.id = self.next_id()

        self.server_log = None
        self.action_queue = action_queue
        self.error_buffer = error_buffer
        self.server_log_buffer = []
        self.enable_logging = enable_logging

        self._broken = False
        self.vnc_client = None
        self.log_manager = None

        self.buf = []
        self.buf_len = 0

        self.queued_data = []
        self.initialized = False

        self.challenge = auth.challenge()
        self.accept_any_password = True
        self.expect(self.recv_ProtocolVersion_Handshake, 12)

    def start_logging(self):
        self.log_manager = LogManager(self.factory.logfile_dir, self.factory.recorder_id, self.id)
        self.server_log = fbs_writer.FBSWriter(self.log_manager.server_logfile)

        for data in self.server_log_buffer:
            self.server_log.write(data)
        # Done with the buffer!
        self.server_log_buffer = None


    def connectionMade(self):
        logger.info('[%s] Connection received from VNC client', self.id)
        factory = protocol.ClientFactory()
        factory.protocol = VNCProxyClient
        factory.vnc_server = self
        factory.deferrable = defer.Deferred()
        endpoint = endpoints.clientFromString(reactor, self.factory.vnc_address)

        def _established_callback(client):
            if self._broken:
                client.close()
            self.vnc_client = client
            self.flush()
        def _established_errback(reason):
            logger.error('[VNCProxyServer] Connection succeeded but could not establish session: %s', reason)
            self.close()
        factory.deferrable.addCallbacks(_established_callback, _established_errback)

        def _connect_errback(reason):
            logger.error('[VNCProxyServer] Connection failed: %s', reason)
            self.close()
        endpoint.connect(factory).addErrback(_connect_errback)

        self.send_ProtocolVersion_Handshake()

    def connectionLost(self, reason):
        logger.info('Losing connection from VNC user')
        if self.vnc_client:
            self.vnc_client.close()

    def dataReceived(self, data):
        pyprofile.incr('vnc_proxy_server.data.sent.messages')
        pyprofile.incr('vnc_proxy_server.data.sent.bytes', len(data))

        self.buf.append(data)
        self.buf_len += len(data)
        self.flush()

    def sendData(self, data):
        if self.server_log is None:
            self.server_log_buffer.append(data)
        else:
            self.server_log.write(data)
        self.transport.write(data)

    def flush(self):
        if self.buf_len < self.expected_len:
            return
        elif self.vnc_client is None and self.action_queue is None:
            return

        buffer = b''.join(self.buf)
        while not self._broken and len(buffer) >= self.expected_len:
            block, buffer = buffer[:self.expected_len], buffer[self.expected_len:]
            self.handle(self.expected, block)

        self.buf[:] = [buffer]
        self.buf_len = len(buffer)

    def handle(self, type, block):
        logger.debug('[%s] Handling: type=%s', self.id, type)
        try:
            self.expected(block, *self.expected_args, **self.expected_kwargs)
        except Exception as e:
            self._error(e)

    def send_ProtocolVersion_Handshake(self):
        self.sendData(b'RFB 003.003\n')

    def recv_ProtocolVersion_Handshake(self, block):
        # Client chooses RFB version
        match = re.search(b'^RFB (\d{3}).(\d{3})\n$', block)
        assert match, 'Block does not match: {!r}'.format(block)
        major = int(match.group(1))
        minor = int(match.group(2))
        self.protocol_version = (major, minor)
        assert major == 3 and minor in (3, 8), 'Unexpected version: {}'.format((major, minor))

        if minor == 3:
            self.send_VNC_Authentication()
        elif minor == 8:
            self.send_SecurityTypes()

    def send_SecurityTypes(self):
        self.sendData(struct.pack('!BB', 1, 2))
        self.expect(self.recv_SecurityTypesResponse, 1)

    def recv_SecurityTypesResponse(self, block):
        (type,) = struct.unpack('!B', block)
        assert type == 2
        self.send_VNC_Authentication()

    def send_VNC_Authentication(self):
        # Now we tell the client to auth with password. (Only do this
        # because the built-in Mac viewer prompts for a password no
        # matter what.)
        self.sendData(struct.pack('!I', 2))
        self.sendData(self.challenge)
        self.expect(self.recv_VNC_Authentication_response, 16)

    def recv_VNC_Authentication_response(self, block):
        expected = auth.challenge_response(self.challenge)
        if block == expected or self.accept_any_password:
            logger.debug('Client authenticated successfully')
            self.send_SecurityResult_Handshake_success()
        else:
            logger.debug('VNC client supplied incorrect password')
            self.send_SecurityResult_Handshake_failed('Your password was incorrect')

    def send_SecurityResult_Handshake_success(self):
        logger.debug('[%s] Send SecurityResult_Handshake_success', self.id)
        self.sendData(struct.pack('!I', 0))
        self.expect(self.recv_ClientInit, 1)

    def send_SecurityResult_Handshake_failed(self, reason):
        self.sendData(struct.pack('!I', 1))
        self.sendData(struct.pack('!I', len(reason)))
        self.sendData(reason)
        self.close()

    def recv_ClientInit(self, block):
        (shared,) = struct.unpack('!B', block)

        ### Now that the server is up and running, we flush
        ### everything.

        # We don't even create log directory until we've successfully
        # established the connection.

        if self.enable_logging and self.factory.logfile_dir:
            # Log in vnc_recorder; don't log in playback
            self.start_logging()
            self.vnc_client.start_logging()

        for data in self.queued_data:
            self.sendData(data)
        self.queued_data = None
        self.initialized = True

        # Listen for messages
        self.expect(self.recv_ClientToServer, 1)

    def recv_ClientToServer(self, block):
        (message_type,) = struct.unpack('!B', block)
        if message_type == 0:
            self.proxyData(block)
            self.expect(self.recv_SetPixelFormat, 19)
        elif message_type == 2:
            # Do not proxy since we want to transform it
            self.expect(self.recv_SetEncodings, 3)
        elif message_type == 3:
            self.proxyData(block)
            self.expect(self.recv_FramebufferUpdateRequest, 9)
        elif message_type == 4:
            self.proxyData(block)
            self.expect(self.recv_KeyEvent, 7)
        elif message_type == 5:
            self.proxyData(block)
            self.expect(self.recv_PointerEvent, 5)
        elif message_type == 6:
            # Do not proxy since we don't support it
            self.expect(self.recv_ClientCutText, 7)
        else:
            assert False, 'Unknown client to server message type received: {}'.format(message_type)

    def recv_SetPixelFormat(self, block):
        self.proxyData(block)

        (server_pixel_format,) = struct.unpack('!xxx16s', block)

        if self.action_queue:
            self.action_queue.set_pixel_format(server_pixel_format)

        # self.vnc_client.framebuffer.apply_format(server_pixel_format)
        self.expect(self.recv_ClientToServer, 1)

    def recv_SetEncodings(self, block):
        # Do not proxy, as we will write our own transformed version
        # of this shortly.

        (number_of_encodings,) = struct.unpack('!xH', block)
        self._handle_SetEncodings(number_of_encodings, [])

    def _handle_SetEncodings(self, number_of_encodings, encodings):
        if number_of_encodings > 0:
            self.expect(self.recv_SetEncodings_encoding_type, 4, number_of_encodings-1, encodings)
        else:
            supported = []
            unsupported = []
            for encoding in encodings:
                if encoding in self.SUPPORTED_ENCODINGS:
                    supported.append(encoding)
                else:
                    unsupported.append(encoding)

            if unsupported:
                logger.info('[%s] Requested %s unsupported encodings: unsupported=%s supported=%s', self.id, len(unsupported), unsupported, supported)

            if self.add_pseudo_cursor_encoding and constants.PSEUDO_CURSOR_ENCODING not in supported:
                logger.info('[%s] Add PSEUDO_CURSOR_ENCODING', self.id)
                supported.append(constants.PSEUDO_CURSOR_ENCODING)

            logger.debug('[%s] Encodings: %s', self.id, supported)
            if self.vnc_client:
                self.vnc_client.send_SetEncodings(supported)
            self.expect(self.recv_ClientToServer, 1)

    def recv_SetEncodings_encoding_type(self, block, number_of_encodings, encodings):
        # Do not proxy, as we will write our own transformed version
        # of this shortly.

        (encoding,) = struct.unpack('!i', block)
        encodings.append(encoding)
        self._handle_SetEncodings(number_of_encodings, encodings)

    def recv_FramebufferUpdateRequest(self, block):
        self.proxyData(block)

        incremental, x, y, width, height = struct.unpack('!BHHHH', block)
        self.expect(self.recv_ClientToServer, 1)

    def recv_KeyEvent(self, block):
        self.proxyData(block)

        down, key = struct.unpack('!BxxI', block)
        if self.action_queue is not None:
            self.action_queue.key_event(key, down)
        self.expect(self.recv_ClientToServer, 1)

    def recv_PointerEvent(self, block):
        self.proxyData(block)

        buttonmask, x, y = struct.unpack('!BHH', block)
        if self.action_queue is not None:
            self.action_queue.pointer_event(x, y, buttonmask)
        self.expect(self.recv_ClientToServer, 1)

    def recv_ClientCutText(self, block):
        # Drop ClientCutText

        (length,) = struct.unpack('!xxxI', block)
        self.expect(self.recv_ClientCutText_value, length)

    def recv_ClientCutText_value(self, block):
        # Drop ClientCutText

        self.expect(self.recv_ClientToServer, 1)

    def expect(self, type, length, *args, **kwargs):
        self.expected = type
        self.expected_len = length
        self.expected_args = args
        self.expected_kwargs = kwargs

    def proxyData(self, data):
        if self.vnc_client:
            self.vnc_client.recvProxyData(data)

    def recvProxyData(self, data):
        """Write data to server"""
        if self.initialized:
            self.sendData(data)
        else:
            self.queued_data.append(data)

    def close(self):
        logger.info('[%s] Closing', self.id)
        self._broken = True
        if self.transport:
            self.transport.loseConnection()
        if self.log_manager:
            self.log_manager.close()
        if self.server_log:
            self.server_log.close()

    def _error(self, e):
        logger.error('[%s] Connection from client aborting with error: %s', self.id, e)
        traceback.print_exc()
        if self.error_buffer:
            self.error_buffer.record(e)
        self.close()

class VNCProxyClient(protocol.Protocol, object):
    def __init__(self):
        self.id = None
        self.vnc_server = None
        self.client_log = None
        # Messages heading for the server, which haven't been written
        # to disk.
        self.client_log_buffer = []
        self.buf = []
        self.buf_len = 0
        self._broken = False

        self.queued_data = []
        self.initialized = False

        self.expect(self.recv_ProtocolVersion_Handshake, 12)

    def start_logging(self):
        self.client_log = fbs_writer.FBSWriter(self.vnc_server.log_manager.client_logfile)
        for data in self.client_log_buffer:
            self.client_log.write(data)
        # Done with the buffer!
        self.client_log_buffer = None

    def connectionMade(self):
        logger.debug('Connection made to VNC server')
        self.vnc_server = self.factory.vnc_server
        self.id = '{}-client'.format(self.vnc_server.id)

    def connectionLost(self, reason):
        logger.info('Losing connection to VNC server')
        if self.vnc_server:
            self.vnc_server.close()

    def proxyData(self, data):
        """Write data to client"""
        assert self.initialized
        self.vnc_server.recvProxyData(data)

    def recvProxyData(self, data):
        """Write data to client"""
        self.sendData(data)

    def sendData(self, data):
        """Write data to server"""
        # Not set up yet
        if self.client_log is None:
            self.client_log_buffer.append(data)
        else:
            self.client_log.write(data)
        self.transport.write(data)

    def dataReceived(self, data):
        if self.expected is None:
            # We're in direct proxy mode
            self.proxyData(data)
            return

        pyprofile.incr('vnc_proxy_server.data.sent.messages')
        pyprofile.incr('vnc_proxy_server.data.sent.bytes', len(data))

        self.buf.append(data)
        self.buf_len += len(data)
        self.flush()

    def flush(self):
        if self.buf_len < self.expected_len:
            return

        buffer = b''.join(self.buf)
        while self.expected is not None and not self._broken and len(buffer) >= self.expected_len:
            block, buffer = buffer[:self.expected_len], buffer[self.expected_len:]
            self.handle(self.expected, block)

        self.buf[:] = [buffer]
        self.buf_len = len(buffer)

        if self.expected is None:
            data = b''.join(self.buf)
            if data != '':
                self.proxyData(data)
            self.buf = []

    def handle(self, type, block):
        logger.debug('[%s] Handling: type=%s', self.id, type)
        try:
            self.expected(block, *self.expected_args, **self.expected_kwargs)
        except Exception as e:
            self._error(e)

    def recv_ProtocolVersion_Handshake(self, block):
        match = re.search(b'^RFB (\d{3}).(\d{3})\n$', block)
        assert match, 'Expected RFB line, but got: {!r}'.format(block)
        major = int(match.group(1))
        minor = int(match.group(2))
        self.sendData(b'RFB 003.003\n')

        self.expect(self.recv_Security_Handshake, 4)

    def recv_Security_Handshake(self, block):
        (auth,) = struct.unpack('!I', block)
        if auth == 0:
            self.expect(self.recv_SecurityResult_Handshake_failed_length, 4)
        elif auth == 1:
            self.send_ClientInit()
        elif auth == 2:
            self.expect(self.recv_VNC_Authentication, 16)
        else:
            assert False, 'Bad auth: {}'.format(auth)

    def recv_VNC_Authentication(self, block):
        response = auth.challenge_response(block)
        self.sendData(response)
        self.expect(self.recv_SecurityResult_Handshake, 4)

    def recv_SecurityResult_Handshake(self, block):
        (result,) = struct.unpack('!xxxB', block)
        if result == 0:
            logger.debug('VNC Auth succeeded')
            self.send_ClientInit()
        elif result == 1:
            logger.debug('VNC Auth failed.')
            # Server optionally can say why
            self.expect(self.recv_SecurityResult_Handshake_failed_length, 4)
        else:
            assert False, 'Bad security result: {}'.format(result)

    def recv_SecurityResult_Handshake_failed_length(self, block):
        (length,) = struct.unpack('!I', block)
        self.expect(self.recv_SecurityResult_Handshake_failed_reason, length)

    def recv_SecurityResult_Handshake_failed_reason(self, block):
        logger.info('Connection to server failed: %s', block)

    def send_ClientInit(self):
        shared = True
        self.sendData(struct.pack('!B', shared))

        ### Now that the session is up and running, we flush
        ### everything and stop parsing.

        # We're up and running!
        logger.info('[%s] Marking server as connected', self.id)
        self.factory.deferrable.callback(self)

        self.initialized = True
        # Flush queue
        for data in self.queued_data:
            self.sendData(data)
        self.queued_data = None

        # And from now on just do a straight proxy
        self.expect(None, None)

    def expect(self, type, length, *args, **kwargs):
        if type is not None:
            logger.debug('Expecting: %s (length=%s)', type.__name__, length)
            assert isinstance(length, int), "Bad length: {}".format(length)

        self.expected = type
        self.expected_len = length
        self.expected_args = args
        self.expected_kwargs = kwargs

    def close(self):
        logger.info('[%s] Closing', self.id)
        self._broken = True
        if self.transport:
            self.transport.loseConnection()
        if self.client_log:
            self.client_log.close()

    def _error(self, e):
        logger.error('[%s] Connection to server aborting with error: %s', self.id, e)
        traceback.print_exc()
        self.close()

    def send_SetEncodings(self, encodings):
        self.sendData(struct.pack("!BxH", 2, len(encodings)))
        for encoding in encodings:
            self.sendData(struct.pack("!i", encoding))
