#include <iostream>
#include <stdint.h>

#include "murmur3.h"

// #define	FORCE_INLINE inline __attribute__((always_inline))

__device__ __forceinline__
uint32_t fmix32 ( uint32_t h )
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}

__device__ __forceinline__
uint32_t rotl32 ( uint32_t x, int8_t r )
{
  return (x << r) | (x >> (32 - r));
}

__global__
void _Murmur3_helper(const void * key, int len, uint32_t seed, void * out) {
    const uint8_t * data = (const uint8_t*)key;
    const int nblocks = len / 4;

    uint32_t h1 = seed;

    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    const uint32_t * blocks = (const uint32_t *)(data + nblocks*4);

    for(int i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];

        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1*5+0xe6546b64;
    }

    const uint8_t * tail = (const uint8_t*)(data + nblocks*4);

    uint32_t k1 = 0;

    switch(len & 3) {
        case 3:
            k1 ^= tail[2] << 16;
        case 2:
            k1 ^= tail[1] << 8;
        case 1:
            k1 ^= tail[0];
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
    };

    //----------
    // finalization

    h1 ^= len;

    h1 = fmix32(h1);

    *(uint32_t*)out = h1;
}

void MurmurHash3_cuda_32(const void * key, int len, uint32_t seed, void * out) {
    void * cuda_key;
    void * cuda_out;

    cudaMalloc(&cuda_key, len);
    cudaMalloc(&cuda_out, sizeof(uint32_t));

    cudaMemcpy(cuda_key, key, len, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_out, out, sizeof(uint32_t), cudaMemcpyHostToDevice);

    _Murmur3_helper<<<1, 1>>>(cuda_key, len, seed, cuda_out);

    cudaDeviceSynchronize();

    cudaMemcpy(out, cuda_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(cuda_key);
    cudaFree(cuda_out);
}
