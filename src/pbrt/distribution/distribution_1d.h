#pragma once

#include <cuda/std/tuple>
#include <pbrt/gpu/macro.h>
#include <vector>

class AliasTable;
class GPUMemoryAllocator;

class Distribution1D {
  public:
    static const Distribution1D *create(const std::vector<FloatType> &values,
                                        GPUMemoryAllocator &allocator);

    void build(const std::vector<FloatType> &values, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    cuda::std::pair<uint, FloatType> sample(FloatType u) const;

    PBRT_CPU_GPU
    FloatType get_pdf(uint idx) const;

  private:
    const AliasTable *alias_table;
};
