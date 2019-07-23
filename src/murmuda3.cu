//
#include <stdio.h>
#include <iostream>
#include <stdint.h>

#include "murmuda3.h"


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

__device__
void _Murmur3_helper_fast(const void * key, int len, uint32_t seed, void * out) {
    const uint8_t * data = (const uint8_t*)key;
    const int nblocks = len / 4;

    uint32_t h1 = seed;

    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    const uint32_t * blocks = (const uint32_t *)(data + nblocks*4);

    uint32_t k1;

    for(int i = -nblocks; i; i++) {
        k1 = blocks[i];

        asm("{\n\t"
            " mul.lo.u32 %0, %0, %1;\n\t"
            ".reg .b32 t1;\n\t"
            ".reg .b32 t2;\n\t"
            " shl.b32 t1, %0, 15;\n\t"
            " shl.b32 t2, %0, 17;\n\t"
            " or.b32 %0, t1, t2;\n\t"
            " mul.lo.u32 %0, %0, %1;\n\t"
            "}"
            : "+r"(k1) : "r"(c1), "r"(c2));

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1*5+0xe6546b64;
    }

    const uint8_t * tail = (const uint8_t*)(data + nblocks*4);

    k1 = 0;

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

__device__
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

__global__
void _batch_helper(const void * keys, int len, int num_keys,
                   uint32_t * seeds, int num_seeds,
                   void * out) {

    const int32_t * data = (const int32_t*)keys;
    uint32_t* out_int = (uint32_t*)out;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int k = index; k < num_keys * num_seeds; k+= stride) {
        int key_index = k / num_seeds;
        int seed_index = k % num_seeds;
        _Murmur3_helper(data + key_index, len,
                        seeds[seed_index],
                        out_int + k);
    }
}


void MurmurHash3_batch(const void * keys, int len, int num_keys,
                       uint32_t * seeds, int num_seeds,
                       void * out) {

    void * cuda_keys;
    uint32_t * cuda_seeds;
    void * cuda_outs;


    cudaMalloc(&cuda_keys, len * num_keys);
    cudaMalloc(&cuda_seeds, num_seeds * sizeof(uint32_t));
    cudaMalloc(&cuda_outs, num_keys * num_seeds * sizeof(uint32_t));

    cudaMemcpy(cuda_keys, keys, len * num_keys, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_seeds, seeds, num_seeds * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_keys * num_seeds + blockSize - 1) / blockSize;

    _batch_helper<<<numBlocks, blockSize>>>(cuda_keys, len, num_keys, cuda_seeds, num_seeds, cuda_outs);

    cudaDeviceSynchronize();

    cudaMemcpy(out, cuda_outs, num_keys * num_seeds * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(cuda_keys);
    cudaFree(cuda_seeds);
    cudaFree(cuda_outs);
}
