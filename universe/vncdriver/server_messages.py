try:
    # In Py2, use the more efficient cStringIO implementation if it's
    # available
    from cStringIO import StringIO as BytesIO
except ImportError:
    # Fall back to using normal BytesIO, six handles python 2 vs 3 compat
    from six import BytesIO

import logging
import numpy as np
from universe import pyprofile
import struct

logger = logging.getLogger(__name__)

class FramebufferUpdate(object):
    def __init__(self, rectangles):
        self.rectangles = rectangles

class Rectangle(object):
    def __init__(self, x, y, width, height, encoding):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.encoding = encoding

class PseudoCursorEncoding(object):
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask

    @classmethod
    def parse_rectangle(cls, client, x, y, width, height, data):
        split = width * height * client.framebuffer.bypp
        image = np.frombuffer(data[:split], np.uint8).reshape((height, width, 4))[:, :, [0, 1, 2]]

        # Turn raw bytes into uint8 array
        mask = np.frombuffer(data[split:], np.uint8)
        # Turn uint8 array into bit array, and go over the scanlines
        mask = np.unpackbits(mask).reshape((height, -1 if mask.size else 0))[:, :width]

        encoding = cls(image, mask)
        return Rectangle(x, y, width, height, encoding)

class RAWEncoding(object):
    def __init__(self, data):
        self.data = data

    @classmethod
    def parse_rectangle(cls, client, x, y, width, height, data):
        assert client.framebuffer.bpp == 32
        data = np.frombuffer(data, np.uint8).reshape((height, width, 4))[:, :, [0, 1, 2]]
        encoding = cls(data)
        return Rectangle(x, y, width, height, encoding)

class ZlibEncoding(object):
    def __init__(self, data):
        self.data = data

    @classmethod
    def parse_rectangle(cls, client, x, y, width, height, data):
        decompressed = client.zlib_decompressor.decompress(data)
        logger.debug('[zlib] Decompressed from %s bytes -> %s bytes', len(data), len(decompressed))
        pyprofile.incr('vncdriver.recv_rectangle.zlib_encoding.decompressed_bytes', len(decompressed), unit=pyprofile.BYTES)
        data = np.frombuffer(decompressed, np.uint8).reshape((height, width, 4))[:, :, [0, 1, 2]]
        encoding = cls(data)
        return Rectangle(x, y, width, height, encoding)

class ZRLEEncoding(object):
    def __init__(self, data):
        self.data = data

    @classmethod
    def parse_rectangle(cls, client, x, y, width, height, data):
        decompressed = client.zlib_decompressor.decompress(data)
        logger.debug('[zrle] Decompressed from %s bytes -> %s bytes', len(data), len(decompressed))
        pyprofile.incr('vncdriver.recv_rectangle.zrle_encoding.decompressed_bytes', len(decompressed), unit=pyprofile.BYTES)

        if client.framebuffer.bpp > 24:
            bytes_per_pixel = 3
        else:
            bytes_per_pixel = client.framebuffer.bypp

        buf = BytesIO(decompressed)
        data = cls._read(x, y, width, height, buf, bytes_per_pixel)
        encoding = cls(data)
        return Rectangle(x, y, width, height, encoding)

    @classmethod
    def _read(cls, x, y, width, height, buf, bytes_per_pixel):
        data = np.zeros([height, width, 3], dtype=np.uint8) + 255

        for tile_y in range(0, height, 64):
            tile_height = min(64, height-tile_y)

            for tile_x in range(0, width, 64):
                tile_width = min(64, width-tile_x)

                tile = data[tile_y:tile_y+tile_height, tile_x:tile_x+tile_width]
                cls._read_tile(tile, buf, tile_width, tile_height, bytes_per_pixel)
        return data

    @classmethod
    def _read_tile(cls, tile, buf, tile_width, tile_height, bytes_per_pixel):
        assert bytes_per_pixel == 3

        # Each tile begins with a subencoding type byte.  The top bit of this
        # byte is set if the tile has been run-length encoded, clear otherwise.
        # The bottom 7 bits indicate the size of the palette used: zero means
        # no palette, 1 means that the tile is of a single color, and 2 to 127
        # indicate a palette of that size.  The special subencoding values 129
        # and 127 indicate that the palette is to be reused from the last tile
        # that had a palette, with and without RLE, respectively.
        (subencoding,) = struct.unpack('!B', buf.read(1))

        run_length_encoded = bool(subencoding & 128)
        palette_size = subencoding & 127

        bytes = palette_size * bytes_per_pixel
        palette_data = buf.read(bytes)
        assert len(palette_data) == bytes, "Palette data came up short: {} bytes rather than {}".format(len(palette_data), bytes)

        logger.debug('Handling zrle tile: run_length_encoded=%s palette_size=%s', run_length_encoded, palette_size)

        if palette_size == 0 and not run_length_encoded:
            # 0: Raw pixel data. width*height pixel values follow (where width and
            # height are the width and height of the tile):
            #
            #  +-----------------------------+--------------+-------------+
            #  | No. of bytes                | Type [Value] | Description |
            #  +-----------------------------+--------------+-------------+
            #  | width*height*BytesPerCPixel | CPIXEL array | pixels      |
            #  +-----------------------------+--------------+-------------+
            data = buf.read(bytes_per_pixel * tile_width * tile_height)
            data = np.frombuffer(data, dtype=np.uint8).reshape((tile_height, tile_width, 3))
            tile[:, :, :] = data
            return
        elif palette_size == 1 and not run_length_encoded:
            # 1: A solid tile consisting of a single color.  The pixel value
            # follows:
            #
            # +----------------+--------------+-------------+
            # | No. of bytes   | Type [Value] | Description |
            # +----------------+--------------+-------------+
            # | bytesPerCPixel | CPIXEL       | pixelValue  |
            # +----------------+--------------+-------------+
            color = np.frombuffer(palette_data, dtype=np.uint8).reshape((3, ))
            tile[:, :, :] = color
            return
        elif not run_length_encoded:
            # 2 to 16:  Packed palette types.  The paletteSize is the value of the
            # subencoding, which is followed by the palette, consisting of
            # paletteSize pixel values.  The packed pixels follow, with each
            # pixel represented as a bit field yielding a zero-based index into
            # the palette.  For paletteSize 2, a 1-bit field is used; for
            # paletteSize 3 or 4, a 2-bit field is used; and for paletteSize
            # from 5 to 16, a 4-bit field is used.  The bit fields are packed
            # into bytes, with the most significant bits representing the
            # leftmost pixel (i.e., big endian).  For tiles not a multiple of 8,
            # 4, or 2 pixels wide (as appropriate), padding bits are used to
            # align each row to an exact number of bytes.

            #   +----------------------------+--------------+--------------+
            #   | No. of bytes               | Type [Value] | Description  |
            #   +----------------------------+--------------+--------------+
            #   | paletteSize*bytesPerCPixel | CPIXEL array | palette      |
            #   | m                          | U8 array     | packedPixels |
            #   +----------------------------+--------------+--------------+

            #  where m is the number of bytes representing the packed pixels.
            #  For paletteSize of 2, this is div(width+7,8)*height; for
            #  paletteSize of 3 or 4, this is div(width+3,4)*height; or for
            #  paletteSize of 5 to 16, this is div(width+1,2)*height.
            palette = np.frombuffer(palette_data, dtype=np.uint8).reshape((-1, 3))

            if palette_size > 16:
                # No palette reuse in zrle
                assert palette_size < 127
                bits_per_packed_pixel = 8
            elif palette_size > 4:
                bits_per_packed_pixel = 4
            elif palette_size > 2:
                bits_per_packed_pixel = 2
            else:
                bits_per_packed_pixel = 1

            for j in range(tile_height):
                # Discard any leftover bits for each new line
                b = 0
                nbits = 0

                for i in range(tile_width):
                    if nbits == 0:
                        (b,) = struct.unpack('!B', buf.read(1))
                        nbits = 8
                    nbits -= bits_per_packed_pixel
                    idx = (b >> nbits) & ((1 << bits_per_packed_pixel) - 1) & 127
                    tile[j, i, :] = palette[idx]
            return
        elif run_length_encoded and palette_size == 0:
            # 128:  Plain RLE.  The data consists of a number of runs, repeated
            # until the tile is done.  Runs may continue from the end of one row
            # to the beginning of the next.  Each run is represented by a single
            # pixel value followed by the length of the run.  The length is
            # represented as one or more bytes.  The length is calculated as one
            # more than the sum of all the bytes representing the length.  Any
            # byte value other than 255 indicates the final byte.  So for
            # example, length 1 is represented as [0], 255 as [254], 256 as
            # [255,0], 257 as [255,1], 510 as [255,254], 511 as [255,255,0], and
            # so on.
            #
            # +-------------------------+--------------+-----------------------+
            # | No. of bytes            | Type [Value] | Description           |
            # +-------------------------+--------------+-----------------------+
            # | bytesPerCPixel          | CPIXEL       | pixelValue            |
            # | div(runLength - 1, 255) | U8 array     | 255                   |
            # | 1                       | U8           | (runLength-1) mod 255 |
            # +-------------------------+--------------+-----------------------+
            i = 0
            pixels = tile_width * tile_height
            data = np.zeros((pixels, 3))
            while i < pixels:
                pix = buf.read(bytes_per_pixel)
                pix = np.frombuffer(pix, dtype=np.uint8).reshape((3, ))

                count = 1
                b = None

                while b == 255 or b is None:
                    (b,) = struct.unpack('!B', buf.read(1))
                    count += b

                data[i:i+count, :] = pix
                i += count
            assert i == pixels
        elif run_length_encoded and palette_size > 1:
            # 130 to 255:  Palette RLE.  Followed by the palette, consisting of
            # paletteSize = (subencoding - 128) pixel values:
            #
            #   +----------------------------+--------------+-------------+
            #   | No. of bytes               | Type [Value] | Description |
            #   +----------------------------+--------------+-------------+
            #   | paletteSize*bytesPerCPixel | CPIXEL array | palette     |
            #   +----------------------------+--------------+-------------+
            #
            # Following the palette is, as with plain RLE, a number of runs,
            # repeated until the tile is done.  A run of length one is
            # represented simply by a palette index:
            #
            #         +--------------+--------------+--------------+
            #         | No. of bytes | Type [Value] | Description  |
            #         +--------------+--------------+--------------+
            #         | 1            | U8           | paletteIndex |
            #         +--------------+--------------+--------------+
            #
            # A run of length more than one is represented by a palette index
            # with the top bit set, followed by the length of the run as for
            # plain RLE.
            #
            # +-------------------------+--------------+-----------------------+
            # | No. of bytes            | Type [Value] | Description           |
            # +-------------------------+--------------+-----------------------+
            # | 1                       | U8           | paletteIndex + 128    |
            # | div(runLength - 1, 255) | U8 array     | 255                   |
            # | 1                       | U8           | (runLength-1) mod 255 |
            # +-------------------------+--------------+-----------------------+
            palette = np.frombuffer(palette_data, dtype=np.uint8).reshape((-1, 3))

            i = 0
            pixels = tile_width * tile_height
            data = np.zeros((pixels, 3))
            while i < pixels:
                (idx,) = struct.unpack('!B', buf.read(1))

                count = 1

                if idx & 128:
                    b = None
                    while b == 255 or b is None:
                        (b,) = struct.unpack('!B', buf.read(1))
                        count += b

                idx &= 127
                pix = palette[idx]

                data[i:i+count, :] = pix
                i += count
            assert i == pixels
        else:
            assert False, "Unhandled case: run_length_encoded={} palette_size={}".format(run_length_encoded, palette_size)

        tile[:] = data.reshape((tile_height, tile_width, 3))
