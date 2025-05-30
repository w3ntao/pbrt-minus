#pragma once

#include <cuda/std/tuple>
#include <pbrt/gpu/macro.h>
#include <vector>

class GPUMemoryAllocator;

class AliasTable {
  public:
    static const AliasTable *create(const std::vector<Real> &values,
                                    GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    cuda::std::pair<int, Real> sample(const Real u0) const;

    struct Bin {
        Real p;
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
        Bin(const Real _p, const int a) : p(_p), first_idx(a), second_idx(-1) {}
    };

    const Bin *bins;
    const Real *pdfs;
    int size;
};
