import heapq
from collections import Counter
import numpy as np
import struct
import pickle


class Node:
    def __init__(self, v, f):
        self.v = v
        self.f = f
        self.l = None
        self.r = None

    def __lt__(self, other):
        return self.f < other.f


def build_tree(freq):
    heap = [Node(v, f) for v, f in freq.items()]
    heapq.heapify(heap)

    # ✅ FIX EDGE CASE: chỉ có 1 symbol
    if len(heap) == 1:
        only = heapq.heappop(heap)
        root = Node(None, only.f)
        root.l = only
        return root

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        parent = Node(None, a.f + b.f)
        parent.l = a
        parent.r = b
        heapq.heappush(heap, parent)

    return heap[0]


def build_codes(node, s="", codes=None):
    if codes is None:
        codes = {}

    if node.v is not None:
        codes[node.v] = s if s != "" else "0"
        return codes

    if node.l:
        build_codes(node.l, s + "0", codes)
    if node.r:
        build_codes(node.r, s + "1", codes)

    return codes


def huffman_encode(data):
    freq = Counter(int(x) for x in data)
    root = build_tree(freq)
    codes = build_codes(root)

    bitstring = "".join(codes[int(x)] for x in data)

    padding = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * padding

    byte_array = bytearray()
    for i in range(0, len(bitstring), 8):
        byte_array.append(int(bitstring[i:i+8], 2))

    return byte_array, freq, padding


def huffman_decode(byte_array, freq, padding):
    root = build_tree(freq)

    bits = ""
    for b in byte_array:
        bits += format(b, "08b")

    if padding > 0:
        bits = bits[:-padding]

    data = []
    node = root

    for bit in bits:
        node = node.l if bit == "0" else node.r
        if node.v is not None:
            data.append(node.v)
            node = root

    return np.array(data, dtype=np.uint8)


def save_huffman_to_file(encoded, freq, padding, shape, filepath):
    with open(filepath, "wb") as f:
        h, w = shape
        f.write(struct.pack("II", h, w))
        f.write(struct.pack("I", padding))
        pickle.dump(freq, f)
        f.write(encoded)


def load_huffman_from_file(filepath):
    with open(filepath, "rb") as f:
        h, w = struct.unpack("II", f.read(8))
        padding = struct.unpack("I", f.read(4))[0]
        freq = pickle.load(f)
        encoded = f.read()

    return encoded, freq, padding, (h, w)