#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/base/camera.h"

class Filter;
class IndependentSampler;
class StratifiedSampler;

class Sampler {
  public:
    enum class Type {
        independent,
        stratified,
    };

    static Sampler *create(const std::string &sampler_type, uint samples_per_pixel,
                           uint total_pixel_num, std::vector<void *> &gpu_dynamic_pointers);

    std::string get_name() const {
        switch (type) {
        case (Type::independent): {
            return "Independent";
        }

        case (Type::stratified): {
            return "Stratified";
        }
        }

        REPORT_FATAL_ERROR();
        return "";
    }

    PBRT_CPU_GPU
    void init(IndependentSampler *independent_sampler);

    PBRT_CPU_GPU
    void init(StratifiedSampler *stratified_sampler);

    PBRT_GPU
    void start_pixel_sample(uint pixel_idx, uint sample_idx, uint dimension);

    PBRT_CPU_GPU
    uint get_samples_per_pixel() const;

    PBRT_GPU FloatType get_1d();

    PBRT_GPU Point2f get_2d();

    PBRT_GPU Point2f get_pixel_2d();

    PBRT_GPU
    CameraSample get_camera_sample(Point2i pPixel, const Filter *filter);

  private:
    Type type;
    void *ptr;
};
