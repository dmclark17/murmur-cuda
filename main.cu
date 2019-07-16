#include <iostream>
#include <stdint.h>

#include "murmur3.h"

int main() {
    int32_t key = 1776;
    int len = sizeof(key);
    uint32_t seed = 10;
    int32_t out;

    MurmurHash3_cuda_32(&key, len, seed, &out);

    std::cout << out << std::endl;
}
