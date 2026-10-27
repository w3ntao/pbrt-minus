#pragma once

#include <pbrt/gpu/macro.h>

PBRT_CPU_GPU
inline int permutation_element(uint32_t i, uint32_t l, uint32_t seed) {
    uint32_t w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;
    do {
        i ^= seed;
        i *= 0xe170893d;
        i ^= seed >> 16;
        i ^= (i & w) >> 4;
        i ^= seed >> 8;
        i *= 0x0929eb3f;
        i ^= seed >> 23;
        i ^= (i & w) >> 1;
        i *= 1 | seed >> 27;
        i *= 0x6935fa69;
        i ^= (i & w) >> 11;
        i *= 0x74dcb303;
        i ^= (i & w) >> 2;
        i *= 0x9e501cc3;
        i ^= (i & w) >> 2;
        i *= 0xc860a3df;
        i &= w;
        i ^= i >> 5;
    } while (i >= l);

    return (uint64_t(i) + seed) % l;
}
