#pragma once

#include <vector>
#include <cuda/std/tuple>

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/util/macro.h"

class Distribution1D;
class GPUImage;

class Distribution2D {
  public:
    static const Distribution2D *create_from_image(const GPUImage *image,
                                                   std::vector<void *> &gpu_dynamic_pointers);

    void build_from_image(const GPUImage *image, std::vector<void *> &gpu_dynamic_pointers);

    PBRT_GPU
    cuda::std::pair<Point2f, FloatType> sample(const Point2f &uv) const;

    PBRT_CPU_GPU
    FloatType get_pdf(const Point2f &u) const;

  private:
    const FloatType *cdf;
    const FloatType *pmf;

    class Distribution1D *distribution_1d_list;

    Point2i dimension;
};
