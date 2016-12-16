import six
import uuid

from universe import utils
from universe.vncdriver.vendor import pydes

class RFBDes(pydes.des):
    def setKey(self, key):
        key = key.encode('ascii')

        newkey = []
        for ki in range(len(key)):
            if six.PY2:
                bsrc = ord(key[ki])
            else:
                bsrc = key[ki]

            # Reverse the bits
            btgt = 0
            for i in range(8):
                if bsrc & (1 << i):
                    btgt = btgt | (1 << 7-i)

            if six.PY2:
                newkey.append(chr(btgt))
            else:
                newkey.append(btgt)

        super(RFBDes, self).setKey(newkey)

def challenge():
    length = 16
    buf = b''
    while len(buf) < length:
        entropy = uuid.uuid4().bytes
        buf += entropy
    return buf[:length]

def challenge_response(challenge, password=None):
    if password is None:
        password = utils.default_password()
    password += ((8 - len(password)) % 8) * '\0'  # pad to multiple of 8 bytes
    des = RFBDes(password)
    return des.encrypt(challenge)
