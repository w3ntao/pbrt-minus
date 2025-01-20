#pragma once

#include <cuda/std/tuple>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/gpu/macro.h>
#include <vector>

class AliasTable;
class Distribution1D;
class GPUMemoryAllocator;

class Distribution2D {
  public:
    static const Distribution2D *create(const std::vector<std::vector<FloatType>> &data,
                                        GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    cuda::std::pair<Point2f, FloatType> sample(const Point2f &uv) const;

    PBRT_CPU_GPU
    FloatType get_pdf(const Point2f &u) const;

  private:
    Point2i dimension;

    Distribution1D *dimension_y_distribution_list;
    const AliasTable *dimension_x_distribution;

    void build(const std::vector<std::vector<FloatType>> &data, GPUMemoryAllocator &allocator);
};
