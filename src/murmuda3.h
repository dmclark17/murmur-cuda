#ifndef _MURMUR3_CUDA_H_
#define _MURMUR3_CUDA_H_

#include <stdint.h>

void MurmurHash3_batch(const void * keys, int len, int num_keys,
                       uint32_t * seeds, int num_seeds,
                       void * out);

#endif // _MURMUR3_CUDA_H_
