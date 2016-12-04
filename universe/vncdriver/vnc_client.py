import collections
try:
    import cStringIO as StringIO
except ImportError:
    from six import StringIO
import logging
import math
import numpy as np
import os
from universe import pyprofile
import re
import struct
import zlib

from twisted.internet import defer, protocol, threads

from universe import utils
from universe.twisty import reactor
from universe.vncdriver import auth, constants, error, screen, server_messages

logger = logging.getLogger(__name__)

# pyprofile.profile.print_frequency = 1
# pyprofile.profile.print_filter = lambda event: event.startswith('vncdriver.recv_rectangle')

def peer_address(peer):
    return '{}:{}'.format(peer.host, peer.port)

class Framebuffer(object):
    def __init__(self, width, height, server_pixel_format, name):
        # self.observer = observer

        self.width = width
        self.height = height
        self.name = name

        self.numpy_screen = screen.NumpyScreen(width, height)
        self.apply_format(server_pixel_format)

    def apply_format(self, server_pixel_format):
        self.server_pixel_format = server_pixel_format
        (self.bpp, self.depth, self.bigendian, self.truecolor,
         self.redmax, self.greenmax, self.bluemax,
         self.redshift, self.greenshift, self.blueshift) = \
           struct.unpack('!BBBBHHHBBBxxx', server_pixel_format)

        # x11vnc will set truecolor == 0 by default. We just have to
        # hope the client will override it.
        #
        # assert self.bigendian == 0
        # Don't want to deal with colormaps...
        # assert self.truecolor == 1

        # Derived values
        self.bypp = self.bpp // 8 # bytes per pixel

        shifts = [self.redshift, self.greenshift, self.blueshift]
        assert set(shifts) == set([0, 8, 16]), 'Surprising pixelformat: {}'.format(self.__dict__)
        # How to cycle pixels from images to get RGB
        self.color_cycle = np.argsort(shifts)
        self.server_init = struct.pack('!HH16sI', self.width, self.height, self.server_pixel_format, len(self.name)) + self.name

        self.numpy_screen.color_cycle = self.color_cycle

class VNCClient(protocol.Protocol, object):
    def __init__(self):
        self.numpy_screen = None

        self.buf = []
        self.buf_len = 0

        self.initialized = False

        # These could be passed around in
        # expected_args/expected_kwargs, but they are needed in so
        # many places it'd become confusing.
        self._remaining_rectangles = None
        self._rectangles = None

        self._pause = False

        self.expect(self.recv_ProtocolVersion_Handshake, 12)

        self._close = False
        self.zlib_decompressor = zlib.decompressobj()

        self._pointer_x = None
        self._pointer_y = None

    def connectionLost(self, reason):
        if not self._close:
            logger.info('Server %s hung up: %s', peer_address(self.transport.getPeer()), reason)
            self._error(error.Error('[{}] Lost connection: {}'.format(self.factory.label, reason)))

    def sendMessage(self, data):
        pyprofile.incr('vnc_client.data.sent.messages')
        pyprofile.incr('vnc_client.data.sent.bytes', len(data), unit=pyprofile.BYTES)
        if self.transport:
            self.transport.write(data)

    def dataReceived(self, data):
        pyprofile.incr('vnc_client.data.received.messages')
        pyprofile.incr('vnc_client.data.received.bytes', len(data), unit=pyprofile.BYTES)

        self.buf.append(data)
        self.buf_len += len(data)
        logger.debug('Received data: %s bytes (brings us to %s total)', len(data), self.buf_len)

        self.flush()

    def flush(self):
        if self.buf_len < self.expected_len:
            return
        elif self._pause:
            # Not strictly needed, but short circuits in case we're
            # paused for a while.
            return
        elif self._close:
            return

        buffer = b''.join(self.buf)
        while len(buffer) >= self.expected_len:
            logger.debug('Remaining in buffer: %s bytes', len(buffer))
            if self._pause:
                logger.debug('Pausing with %s bytes left in the buffer', len(buffer))
                break
            block, buffer = buffer[:self.expected_len], buffer[self.expected_len:]
            if not self.handle(self.expected, block):
                logger.debug('Stopping due to error in handle()')
                buffer = block + buffer
                break

        self.buf[:] = [buffer]
        self.buf_len = len(buffer)

    def handle(self, type, block):
        logger.debug('Handling server event: type=%s', type.__name__)
        try:
            self.expected(block, *self.expected_args, **self.expected_kwargs)
            return True
        except Exception as e:
            self._error(e)
            return False

    def recv_ProtocolVersion_Handshake(self, block):
        match = re.search(b'^RFB (\d{3}).(\d{3})\n$', block)
        assert match, 'Expected RFB line, but got: {!r}'.format(block)
        major = int(match.group(1))
        minor = int(match.group(2))
        self.sendMessage(b'RFB 003.003\n')

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

    def recv_VNC_Authentication(self, block):
        response = auth.challenge_response(block)
        self.sendMessage(response)
        self.expect(self.recv_SecurityResult_Handshake, 4)

    def send_ClientInit(self):
        shared = True
        self.sendMessage(struct.pack('!B', shared))
        self.expect(self.recv_ServerInit, 24)

    def recv_ServerInit(self, block):
        # This can be rewritten, so don't proxy immediately
        (width, height, server_pixel_format, namelen) = struct.unpack('!HH16sI', block)
        self.expect(self.recv_ServerInit_name, namelen, width, height, server_pixel_format)

    def recv_ServerInit_name(self, block, width, height, server_pixel_format):
        self.framebuffer = Framebuffer(width, height, server_pixel_format, block)
        self.numpy_screen = self.framebuffer.numpy_screen

        self.initialized = True
        self.send_PixelFormat()
        self.send_SetEncodings([
            constants.ZRLE_ENCODING,
            constants.ZLIB_ENCODING,
            constants.RAW_ENCODING,
        ])
        self.send_FramebufferUpdateRequest(incremental=1)

        if self.factory.deferred:
            self.factory.deferred.callback(self)

        # All connected!
        self.expect(self.recv_ServerToClient, 1)

    def recv_ServerToClient(self, block):
        (message_type,) = struct.unpack('!B', block)
        if message_type == 0:
            # self.observer.server_data_block(block)
            self.expect(self.recv_FramebufferUpdate, 3)
        elif message_type == 1:
            # self.observer.server_data_block(block)
            self.expect(self.recv_SetColorMapEntries, 5)
        elif message_type == 2:
            # self.observer.server_data_block(block)
            # self.observer.server_bell()
            self.expect(self.recv_ServerToClient, 1)
        elif message_type == 3:
            # Just drop ServerCutText messages
            self.expect(self.recv_ServerCutText, 7)
        else:
            assert False, 'Unknown server to client message type received: {}'.format(message_type)

    def recv_FramebufferUpdate(self, block):
        (number_of_rectangles, ) = struct.unpack('!xH', block)
        logger.debug('Receiving %d rectangles', number_of_rectangles)
        pyprofile.incr('vncdriver.framebuffer_update')
        pyprofile.incr('vncdriver.framebuffer_update.number_of_rectangles', number_of_rectangles)
        self._remaining_rectangles = number_of_rectangles
        self._rectangles = []

        self._process_rectangles()

    def _process_rectangles(self):
        if self._remaining_rectangles > 0:
            self.expect(self.recv_Rectangle, 12)
        else:
            framebuffer_update = server_messages.FramebufferUpdate(self._rectangles)
            self.numpy_screen.apply(framebuffer_update)
            self._remaining_rectangles = None
            self._rectangles = None

            # Once you're done, send another framebufferUpdateRequest.
            #
            # Empirically, after a few framebufferUpdateRequest's without
            # any changes, Xvnc will hold off on sending another
            # framebuffer update until the display data changes.
            self.send_FramebufferUpdateRequest(incremental=1)
            self.expect(self.recv_ServerToClient, 1)

    def recv_Rectangle(self, block):
        self._remaining_rectangles -= 1

        (x, y, width, height, encoding) = struct.unpack('!HHHHi', block)

        if encoding == constants.RAW_ENCODING:
            self.expect(self.recv_DecodeRAW, width*height*self.framebuffer.bypp, x, y, width, height)
        elif encoding == constants.ZRLE_ENCODING:
            self.expect(self.recv_DecodeZRLE, 4, x, y, width, height)
        elif encoding == constants.ZLIB_ENCODING:
            self.expect(self.recv_DecodeZlib, 4, x, y, width, height)
        elif encoding == constants.PSEUDO_CURSOR_ENCODING:
            length = width * height * self.framebuffer.bypp
            length += int(math.floor((width + 7.0) / 8)) * height
            self.expect(self.recv_DecodePseudoCursor, length, x, y, width, height)
        else:
            assert False, 'Unknown pixel encoding received: {}'.format(encoding)

    # Encodings

    def recv_DecodePseudoCursor(self, block, x, y, width, height):
        # cf https://github.com/sibson/vncdotool/blob/master/vncdotool/rfb.py
        rectangle = server_messages.PseudoCursorEncoding.parse_rectangle(self, x, y, width, height, block)
        self._rectangles.append(rectangle)
        self._process_rectangles()

    def recv_DecodeRAW(self, block, x, y, width, height):
        pyprofile.incr('vncdriver.recv_rectangle.raw_encoding')
        pyprofile.incr('vncdriver.recv_rectangle.raw_encoding.bytes', len(block), unit=pyprofile.BYTES)
        rectangle = server_messages.RAWEncoding.parse_rectangle(self, x, y, width, height, block)
        self._rectangles.append(rectangle)
        self._process_rectangles()

    def recv_DecodeZRLE(self, block, x, y, width, height):
        pyprofile.incr('vncdriver.recv_rectangle.zrle_encoding')
        pyprofile.incr('vncdriver.recv_rectangle.zrle_encoding.bytes', len(block), unit=pyprofile.BYTES)

        (length,) = struct.unpack('!I', block)
        self.expect(self.recv_DecodeZRLE_value, length, x, y, width, height)

    def recv_DecodeZRLE_value(self, block, x, y, width, height):
        pyprofile.incr('vncdriver.recv_rectangle.zrle_encoding.bytes', len(block), unit=pyprofile.BYTES)
        rectangle = server_messages.ZRLEEncoding.parse_rectangle(self, x, y, width, height, block)
        self._rectangles.append(rectangle)
        self._process_rectangles()

    def recv_DecodeZlib(self, block, x, y, width, height):
        pyprofile.incr('vncdriver.recv_rectangle.zlib_encoding')
        pyprofile.incr('vncdriver.recv_rectangle.zlib_encoding.bytes', len(block), unit=pyprofile.BYTES)

        (length,) = struct.unpack('!I', block)
        self.expect(self.recv_DecodeZlib_value, length, x, y, width, height)

    def recv_DecodeZlib_value(self, block, x, y, width, height):
        pyprofile.incr('vncdriver.recv_rectangle.zlib_encoding.bytes', len(block), unit=pyprofile.BYTES)
        rectangle = server_messages.ZlibEncoding.parse_rectangle(self, x, y, width, height, block)
        self._rectangles.append(rectangle)
        self._process_rectangles()

    def recv_SetColorMapEntries(self, block):
        # self.observer.server_data_block(block)

        (first_color, number_of_colors) = struct.unpack('!xHH', block)
        self._handle_SetColorMapEntries(first_color, number_of_colors, [])

    def _handle_SetColorMapEntries(self, first_color, number_of_colors, colors):
        if number_of_colors > 0:
            self.expect(self.recv_SetColorMapEntries_color, 6, first_color, number_of_colors-1, colors)
        else:
            # self.observer.server_set_color_map_entries(first_color, colors)
            self.expect(self.recv_ServerToClient, 1)

    def recv_SetColorMapEntries_color(self, block, first_color, number_of_colors, colors):
        # self.observer.server_data_block(block)

        red, green, blue = struct.unpack('!HHH', block)
        colors.append((red, green, blue))
        self._handle_SetColorMapEntries(first_color, number_of_colors, colors)

    def recv_ServerCutText(self, block):
        # We drop these messages
        (length,) = struct.unpack('!xxxI', block)
        self.expect(self.recv_ServerCutText_value, length)

    def recv_ServerCutText_value(self, block):
        # We drop these messages
        # self.observer.server_cut_text(block)
        self.expect(self.recv_ServerToClient, 1)

    def send_SetEncodings(self, encodings):
        self.sendMessage(struct.pack("!BxH", 2, len(encodings)))
        for encoding in encodings:
            self.sendMessage(struct.pack("!i", encoding))

    def send_PixelFormat(self, bpp=32, depth=24, bigendian=0, truecolor=1, redmax=255, greenmax=255, bluemax=255, redshift=0, greenshift=8, blueshift=16):
        if not self.initialized:
            # Not too bad to add, but no need right now. (We'd need to
            # make sure the framebuffer settings don't get
            # overridden.)
            raise error.Error('Framebuffer not initialized. We have not yet added support for queuing PixelFormat messages before initialization')
        server_pixel_format = struct.pack("!BBBBHHHBBBxxx", bpp, depth, bigendian, truecolor, redmax, greenmax, bluemax, redshift, greenshift, blueshift)
        self.sendMessage(struct.pack("!Bxxx16s", 0, server_pixel_format))
        self.framebuffer.apply_format(server_pixel_format)

    def send_FramebufferUpdateRequest(self, x=0, y=0, width=None, height=None, incremental=0):
        if not self.initialized:
            # Not too bad to add, but no need right now. (We'd need to
            # calculate the message after the framebuffer is
            # initialized.)
            raise error.Error('Framebuffer not initialized. We have not yet added support for queuing FramebufferUpdateRequest messages before initialization')
        if width is None:
            width  = self.framebuffer.width - x
        if height is None:
            height = self.framebuffer.height - y
        self.sendMessage(struct.pack("!BBHHHH", 3, incremental, x, y, width, height))

    def send_KeyEvent(self, key, down):
        """For most ordinary keys, the "keysym" is the same as the
        corresponding ASCII value.  Other common keys are shown in the
        KEY_ constants.
        """
        self.sendMessage(struct.pack('!BBxxI', 4, down, key))

    def send_PointerEvent(self, x, y, buttonmask=0):
        """Indicates either pointer movement or a pointer button press or
           release. The pointer is now at (x-position, y-position),
           and the current state of buttons 1 to 8 are represented by
           bits 0 to 7 of button-mask respectively, 0 meaning up, 1
           meaning down (pressed).
        """
        self.sendMessage(struct.pack('!BBHH', 5, buttonmask, x, y))

    def send_ClientCutText(self, message):
        """The client has new text in its clipboard.
        """
        self.sendMessage(struct.pack("!BxxxI", 6, len(message)))
        self.sendMessage(message)

    def expect(self, type, length, *args, **kwargs):
        logger.debug('Expecting: %s (length=%s)', type.__name__, length)
        assert isinstance(length, int), "Bad length (not an int): {}".format(length)

        self.expected = type
        self.expected_len = length
        self.expected_args = args
        self.expected_kwargs = kwargs

    def close(self):
        self._close = True
        if self.transport:
            self.transport.loseConnection()

    def _error(self, e):
        self.close()
        self.factory.error_buffer.record(e)
        if self.factory.deferred:
            try:
                self.factory.deferred.errback(utils.format_error(e))
            except defer.AlreadyCalledError:
                pass

def client_factory(deferred, error_buffer):
    factory = protocol.ClientFactory()
    factory.deferred = deferred
    factory.error_buffer = error_buffer
    factory.protocol = VNCClient
    return factory
