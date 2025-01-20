#pragma once

#include <cuda/std/tuple>
#include <pbrt/gpu/macro.h>
#include <vector>

class GPUMemoryAllocator;

class AliasTable {
  public:
    static const AliasTable *create(const std::vector<FloatType> &values,
                                    GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    cuda::std::pair<uint, FloatType> sample(const FloatType u0) const;

    struct Bin {
        FloatType p;
        int first_idx;
        int second_idx;

        PBRT_CPU_GPU
        bool operator<(const Bin &right) const {
            return p < right.p;
        }

        PBRT_CPU_GPU
        bool operator>(const Bin &right) const {
            return p > right.p;
        }

        PBRT_CPU_GPU
        Bin() : p(NAN), first_idx(-1), second_idx(-1) {}

        PBRT_CPU_GPU
        Bin(const FloatType _p, const int a) : p(_p), first_idx(a), second_idx(-1) {}
    };

    const Bin *bins;
    const FloatType *pdfs;
    uint size;
};
