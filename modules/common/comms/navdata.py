import struct

from modules.common.comms.baseclass import BaseClass
from modules.common.comms.definitions import MessageType


__author__ = "EnriqueMoran"


MIN_ID = 0
MAX_ID = 0
MIN_DIST = 0
MAX_DIST = 18520
MIN_BEARING = 0
MAX_BEARING = 360


class NavData(BaseClass):

    def __init__(self):
        super().__init__()
        self._id = 0
        self._distance = 0
        self._bearing  = 0
        self._message_type = MessageType.NAV_DATA

    def pack(self):
        """
        type     -> Type: Int   | Num Bytes: 1 | Endianness: Big-endian
        id       -> Type: Int   | Num Bytes: 1 | Endianness: Big-endian
        distance -> Type: Float | Num Bytes: 4 | Endianness: Little-endian
        bearing  -> Type: Int   | Num Bytes: 2 | Endianness: Big-endian
        """
        res = struct.pack('>B', self._message_type.value) + \
              struct.pack('>B', self._id)                 + \
              struct.pack('<f', self._distance)           + \
              struct.pack('>H', int(self._bearing))
        return res

    @staticmethod
    def unpack(packed_data):
        """
        type     -> Type: Int   | Num Bytes: 1 | Endianness: Big-endian
        id       -> Type: Int   | Num Bytes: 1 | Endianness: Big-endian
        distance -> Type: Float | Num Bytes: 4 | Endianness: Little-endian
        bearing  -> Type: Int   | Num Bytes: 2 | Endianness: Big-endian
        """
        message_type = MessageType(struct.unpack('>B', packed_data[0:1])[0])
        id_unpacked  = struct.unpack('>B', packed_data[1:2])[0]
        distance_unpacked = struct.unpack('<f', packed_data[2:6])[0]
        bearing_unpacked  = struct.unpack('>H', packed_data[6:8])[0]

        return {
            'type': message_type,
            'id': id_unpacked,
            'distance': distance_unpacked,
            'bearing': bearing_unpacked
        }


    @property
    def type(self):
        return self._message_type
    

    @property
    def id(self):
        return self._id
    

    @id.setter
    def id(self, new_id):
        if isinstance(new_id, int) and new_id >= MIN_ID and new_id <= MAX_ID:
            self._x = new_id
        else:
            msg = f"Id ({new_id}) must be an int between {MIN_ID} and {MAX_ID}."
            raise TypeError(msg)
    

    @property
    def distance(self):
        return self._distance
    

    @distance.setter
    def distance(self, new_dist):
        if isinstance(new_dist, float) and new_dist >= MIN_DIST and new_dist <= MAX_DIST:
            self._distance = new_dist
        else:
            msg = f"Distance ({new_dist}) must be a float between {MIN_DIST} and {MAX_DIST}."
            raise TypeError(msg)
    

    @property
    def bearing(self):
        return self._bearing
    

    @bearing.setter
    def bearing(self, new_bear):
        if isinstance(new_bear, int) and new_bear >= MIN_BEARING and new_bear <= MAX_BEARING:
            self._x = new_bear
        else:
            msg = f"Bearing ({new_bear}) must be an int between {MIN_BEARING} and {MAX_BEARING}."
            raise TypeError(msg)