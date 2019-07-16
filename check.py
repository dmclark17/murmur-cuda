import mmh3

key = 1776
seed = 10

byte_key = key.to_bytes(4, byteorder='little', signed=False)

print(byte_key)
print(int.from_bytes(byte_key, byteorder='little', signed=False))

hash = mmh3.hash(byte_key, seed, signed=False)

print(hash)
