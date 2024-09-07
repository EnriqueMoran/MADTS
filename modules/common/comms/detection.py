import struct

from modules.common.comms.baseclass import BaseClass
from modules.common.comms.definitions import MessageType


__author__ = "EnriqueMoran"


MIN_X = 0
MAX_X = 1
MIN_Y = 0
MAX_Y = 1
MIN_WIDTH = 0
MAX_WIDTH = 1
MIN_HEIGHT = 0
MAX_HEIGHT = 1
MIN_PROB = 0
MAX_PROB = 1


class Detection(BaseClass):

    def __init__(self):
        super().__init__()
        self._x = 0
        self._y = 0
        self._width  = 0
        self._height = 0
        self._probability = 0
        self._message_type = MessageType.DETECTION


    def pack(self):
        """
        type        -> Type: Int   | Num Bytes: 1 | Endianness: Big-endian
        x           -> Type: Float | Num Bytes: 4 | Endianness: Little-endian
        y           -> Type: Float | Num Bytes: 4 | Endianness: Little-endian
        width       -> Type: Float | Num Bytes: 4 | Endianness: Little-endian
        height      -> Type: Float | Num Bytes: 4 | Endianness: Little-endian
        probability -> Type: Float | Num Bytes: 4 | Endianness: Little-endian
        """
        res = struct.pack('>B', self._message_type) + \
              struct.pack('<f', self._x)            + \
              struct.pack('<f', self._y)            + \
              struct.pack('<f', self._width)        + \
              struct.pack('<f', self._height)       + \
              struct.pack('<f', self._probability)
        return res
    

    @staticmethod
    def unpack(packed_data):
        """
        type        -> Type: Int   | Num Bytes: 1 | Endianness: Big-endian
        x           -> Type: Float | Num Bytes: 4 | Endianness: Little-endian
        y           -> Type: Float | Num Bytes: 4 | Endianness: Little-endian
        width       -> Type: Float | Num Bytes: 4 | Endianness: Little-endian
        height      -> Type: Float | Num Bytes: 4 | Endianness: Little-endian
        probability -> Type: Float | Num Bytes: 4 | Endianness: Little-endian
        """
        message_type  = MessageType(struct.unpack('>B', packed_data[0:1])[0])
        unpacked_data = struct.unpack('<fffff', packed_data[1:])

        return {
            'type': message_type,
            'x': unpacked_data[0],
            'y': unpacked_data[1],
            'width': unpacked_data[2],
            'height': unpacked_data[3],
            'probability': unpacked_data[4]
        }


    @property
    def type(self):
        return self._message_type
    

    @property
    def x(self):
        return self._x
    

    @x.setter
    def x(self, new_x):
        if isinstance(new_x, float) and new_x >= MIN_X and new_x <= MAX_X:
            self._x = new_x
        else:
            msg = f"X ({new_x}) must be a float between {MIN_X} and {MAX_X}."
            raise TypeError(msg)
    

    @property
    def y(self):
        return self._y
    

    @y.setter
    def y(self, new_y):
        if isinstance(new_y, float) and new_y >= MIN_Y and new_y <= MAX_Y:
            self._y = new_y
        else:
            msg = f"Y ({new_y}) must be a float between {MIN_Y} and {MAX_Y}."
            raise TypeError(msg)
    

    @property
    def width(self):
        return self._width
    

    @width.setter
    def width(self, new_width):
        if isinstance(new_width, float) and new_width >= MIN_WIDTH and new_width <= MAX_WIDTH:
            self._width = new_width
        else:
            msg = f"Width ({new_width}) must be a float between {MIN_WIDTH} and {MAX_WIDTH}."
            raise TypeError(msg)
    

    @property
    def height(self):
        return self._height
    

    @height.setter
    def height(self, new_height):
        if isinstance(new_height, float) and new_height >= MIN_HEIGHT and new_height <= MAX_HEIGHT:
            self._height = new_height
        else:
            msg = f"Height ({new_height}) must be a float between {MIN_HEIGHT} and {MAX_HEIGHT}."
            raise TypeError(msg)
    

    @property
    def probability(self):
        return self._probability
    

    @probability.setter
    def probability(self, new_prob):
        if isinstance(new_prob, float) and new_prob >= MIN_PROB and new_prob <= MAX_PROB:
            self._probability = new_prob
        else:
            msg = f"Probability ({new_prob}) must be a float between {MIN_PROB} and {MAX_PROB}."
            raise TypeError(msg)
    