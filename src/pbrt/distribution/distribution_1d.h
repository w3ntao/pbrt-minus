#pragma once

#include <cuda/std/tuple>
#include <pbrt/gpu/macro.h>
#include <vector>

class AliasTable;
class GPUMemoryAllocator;

class Distribution1D {
  public:
    static const Distribution1D *create(const std::vector<Real> &values,
                                        GPUMemoryAllocator &allocator);

    void build(const std::vector<Real> &values, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    cuda::std::pair<uint, Real> sample(Real u) const;

    PBRT_CPU_GPU
    Real get_pdf(uint idx) const;

  private:
    const AliasTable *alias_table;
};
