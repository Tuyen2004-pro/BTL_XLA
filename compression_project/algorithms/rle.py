import numpy as np
import struct


def rle_encode(data):
    if len(data) == 0:
        return []

    result = []
    prev = int(data[0])
    count = 1

    for x in data[1:]:
        x = int(x)
        if x == prev and count < 255:
            count += 1
        else:
            result.append((prev, count))
            prev = x
            count = 1

    result.append((prev, count))
    return result


def rle_decode(code):
    out = []
    for val, cnt in code:
        out.extend([val] * cnt)
    return np.array(out, dtype=np.uint8)


def save_rle_to_file(code, shape, filepath):
    with open(filepath, "wb") as f:
        h, w = shape
        f.write(struct.pack("II", h, w))

        for val, cnt in code:
            f.write(struct.pack("BB", int(val), int(cnt)))


def load_rle_from_file(filepath):
    with open(filepath, "rb") as f:
        h, w = struct.unpack("II", f.read(8))
        code = []

        while True:
            bytes_read = f.read(2)
            if not bytes_read:
                break
            val, cnt = struct.unpack("BB", bytes_read)
            code.append((val, cnt))

    return code, (h, w)