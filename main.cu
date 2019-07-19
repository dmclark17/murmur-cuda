#include <iostream>
#include <stdint.h>

#include "murmur3.h"

int main() {
    int num_keys = 1000;
    int num_seeds = 100;

    int32_t* keys = new int32_t[num_keys];
    int len = sizeof(keys[0]);
    for (int i = 0; i < num_keys; i++) {
        keys[i] = i;
    }

    uint32_t* seeds = new uint32_t[num_keys];
    for (int i = 0; i < num_seeds; i++) {
        seeds[i] = i;
    }

    uint32_t* out = new uint32_t[num_keys * num_seeds];

    MurmurHash3_batch(keys, len, num_keys, seeds, num_seeds, out);


    // for (int i = 0; i < num_keys; i++) {
    //     for (int j = 0; j < num_seeds; j++) {
    //         std::cout << out[j + i * num_seeds] << std::endl;
    //     }
    // }
}
