#ifndef _MURMUR3_CUDA_H_
#define _MURMUR3_CUDA_H_

#include <stdint.h>

__device__
void _Murmur3_helper(const void * key, int len, uint32_t seed, void * out);

void MurmurHash3_batch(const void * keys, int len, int num_keys,
                       uint32_t * seeds, int num_seeds,
                       void * out);

#endif // _MURMUR3_CUDA_H_
