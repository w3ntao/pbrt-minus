#pragma once

#include <cuda/std/tuple>
#include <pbrt/gpu/macro.h>
#include <vector>

class GPUMemoryAllocator;

class AliasTable {
  public:
    struct Bin {
        Real p = NAN;
        int first_idx = -1;
        int second_idx = -1;

        PBRT_CPU_GPU
        bool operator<(const Bin &right) const {
            return p < right.p;
        }

        PBRT_CPU_GPU
        bool operator>(const Bin &right) const {
            return p > right.p;
        }

        PBRT_CPU_GPU
        Bin() {}

        PBRT_CPU_GPU
        Bin(const Real _p, const int a) : p(_p), first_idx(a) {}
    };

    AliasTable(const std::vector<Real> &values, GPUMemoryAllocator &allocator);

    AliasTable(const Bin *_bins, const Real *_pdfs, const int _size)
        : bins(_bins), pdfs(_pdfs), size(_size) {}

    PBRT_CPU_GPU
    cuda::std::pair<int, Real> sample(Real u0) const;

    const Bin *bins = nullptr;
    const Real *pdfs = nullptr;
    const int size = 0;
};
