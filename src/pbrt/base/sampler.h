#pragma once

#include <pbrt/euclidean_space/point2.h>

class Filter;
class GPUMemoryAllocator;
class IndependentSampler;
class MLTSampler;
class StratifiedSampler;
struct CameraSample;

class Sampler {
  public:
    enum class Type {
        independent,
        mlt,
        stratified,
    };

    static Type parse_sampler_type(const std::string &sampler_type) {
        if (sampler_type == "independent") {
            return Type::independent;
        }

        if (sampler_type == "mlt") {
            return Type::mlt;
        }

        if (sampler_type == "stratified") {
            return Type::stratified;
        }

        REPORT_FATAL_ERROR();
        return Type::independent;
    }

    static Sampler *create_samplers(const std::string &string_sampler_type, uint samples_per_pixel,
                                    uint size, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    void init(IndependentSampler *independent_sampler);

    PBRT_CPU_GPU
    void init(MLTSampler *mlt_sampler);

    PBRT_CPU_GPU
    void init(StratifiedSampler *stratified_sampler);

    PBRT_CPU_GPU
    MLTSampler *get_mlt_sampler() const;

    PBRT_CPU_GPU
    void start_pixel_sample(uint pixel_idx, uint sample_idx, uint dimension);

    PBRT_CPU_GPU
    uint get_samples_per_pixel() const;

    PBRT_CPU_GPU
    Real get_1d();

    PBRT_CPU_GPU
    Point2f get_2d();

    PBRT_CPU_GPU
    Point2f get_pixel_2d();

    PBRT_CPU_GPU
    CameraSample get_camera_sample(Point2i pPixel, const Filter *filter);

  private:
    Type type;
    void *ptr;
};
