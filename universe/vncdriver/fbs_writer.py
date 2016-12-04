import json
import struct
import time

from gym.utils import atomic_write, closer

fbs_closer = closer.Closer()

class FBSWriter(object):
    def __init__(self, path):
        self._closed = False

        self.start = None
        self.stop = None

        self._id = fbs_closer.register(self)

        self.file = open(path, 'wb')
        # custom format: exactly the same as FBS 001.000 except:
        #
        # FBS 001.002
        # {line-of-json}
        # [length-byte, data, timestamp]...
        # \0\0\0\0 {line-of-json}
        self.file.write(b'FBS 001.002\n')

    def write(self, data):
        # Format:
        #
        # length
        # data
        # timestamp (4 bytes)

        if not data:
            return

        if self.start is not None:
            delta = int(1000 * (time.time() - self.start))
        else:
            delta = 0
            self.start = time.time()

            # Write metadata header
            self.file.write(json.dumps({'start': self.start}).encode('utf-8'))
            self.file.write(b'\n')

        length = struct.pack('!I', len(data))
        self.file.write(length)
        self.file.write(data)

        delta = struct.pack('!I', delta)
        self.file.write(delta)

    def _write_metadata(self):
        # Write metadata trailer
        null = struct.pack('!I', 0)
        self.file.write(null)
        self.file.write(json.dumps({'stop': self.stop}).encode('utf-8'))
        self.file.write(b'\n')

    def close(self):
        if self._closed:
            return
        self._closed = True

        fbs_closer.unregister(self._id)
        self.stop = time.time()
        self._write_metadata()
        self.file.close()

    def __del__(self):
        self.close()
