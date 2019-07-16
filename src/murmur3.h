#ifndef _MURMUR3_CUDA_H_
#define _MURMUR3_CUDA_H_

#include <stdint.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y);

__global__
void _Murmur3_helper(const void * key, int len, uint32_t seed, void * out);

void MurmurHash3_cuda_32(const void * key, int len, uint32_t seed, void * out);

#endif // _MURMUR3_CUDA_H_
