import itertools

import mmh3


keys = [1776, 420]
seeds = [10, 11]

for key, seed in itertools.product(keys, seeds):

    byte_key = key.to_bytes(4, byteorder='little', signed=False)

    # print(byte_key)
    # print(int.from_bytes(byte_key, byteorder='little', signed=False))

    hash = mmh3.hash(byte_key, seed, signed=False)

    print(key, seed, hash)
