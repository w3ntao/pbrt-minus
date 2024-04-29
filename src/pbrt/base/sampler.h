#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/base/camera.h"

class Filter;
class IndependentSampler;

class Sampler {
  public:
    PBRT_CPU_GPU
    void init(IndependentSampler *independent_sampler);

    PBRT_GPU
    void start_pixel_sample(uint pixel_idx, uint sample_idx, uint dimension);

    PBRT_GPU FloatType get_1d();

    PBRT_GPU Point2f get_2d();

    PBRT_GPU Point2f get_pixel_2d();

    PBRT_GPU
    CameraSample get_camera_sample(Point2i pPixel, const Filter *filter);

  private:
    enum class SamplerType {
        independent_sampler,
    };

    SamplerType sampler_type;
    void *sampler_ptr;
};
