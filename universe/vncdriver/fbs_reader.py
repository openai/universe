import json
import os
import struct

from universe import error

class InvalidFBSFileError(error.Error):
    pass

class FBSReader(object):
    def __init__(self, path):
        self.file = open(path, 'rb')
        version = self.file.read(12)
        if version != b'FBS 001.002\n':
            raise InvalidFBSFileError('Unrecognized FBS version: {}'.format(version))

        header = self.file.readline()
        pos = self.file.tell()
        self.file.seek(pos, os.SEEK_SET)

        header = json.loads(header.decode('utf-8'))
        self.start = header['start']

    def __iter__(self):
        return self

    def read_safe(self, size=None):
        """
        We currently close our fbs files by killing them, so sometimes they end
        up with bad data at the end. Close our reader if we expect `size` bytes
        and get fewer.

        This is a hack and should be removed when we cleanly close our
        connections in fbs_writer.

        https://github.com/openai/universe-envs/issues/41
        """
        bytes = self.file.read(size)
        if len(bytes) != size:
            # We unexpectedly got to the end of the file
            self.close()
            raise StopIteration
        return bytes

    def next(self):
        return self.__next__()

    def __next__(self):
        length_str = self.read_safe(4)
        if length_str == '':
            # Indicates a file with no trailer
            self.close()
            raise StopIteration
        (length,) = struct.unpack('!I', length_str)

        if length == 0:
            # Reached the end
            self.close()
            raise StopIteration()

        data = self.read_safe(length)
        timestamp_str = self.read_safe(4)
        (timestamp,) = struct.unpack('!I', timestamp_str)

        return data, self.start + timestamp/1000.

    def close(self):
        self.file.close()
