#include <iostream>
#include <stdint.h>

#include "murmuda3.h"
#include "MurmurHash3.h"

int main() {

    int num_keys = 5000;
    int num_seeds = 500;

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

    // Check against the original C implementation from smhasher
    uint32_t* c_out = new uint32_t[1];
    for (int i = 0; i < num_keys; i++) {
        for (int j = 0; j < num_seeds; j++) {
            MurmurHash3_x86_32(&keys[i], len, seeds[j], c_out);
            if (out[j + i * num_seeds] != *c_out) {
                std::cout << "error for key:" << keys[i] << " seed:" <<
                    seeds[j] << std::endl;
            }
        }
    }
}
