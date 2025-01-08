#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/util/macro.h"
#include <vector>

class Distribution1D;

class Distribution2D {
  public:
    static const Distribution2D *create(const std::vector<std::vector<FloatType>> &data,
                                        std::vector<void *> &gpu_dynamic_pointers);

    void build(const std::vector<std::vector<FloatType>> &data,
               std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    std::pair<Point2f, FloatType> sample(const Point2f &uv) const;

    PBRT_CPU_GPU
    FloatType get_pdf(const Point2f &u) const;

  private:
    const FloatType *cdf;
    const FloatType *pmf;

    Distribution1D *distribution_1d_list;

    Point2i dimension;
};
