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
    enum class Type { independent, mlt, stratified };

    static const Type default_sampler_type = Type::stratified;

    static Type sampler_string_to_type(const std::string &sampler_name);

    static std::string sampler_type_to_string(Type sampler_type);

    PBRT_CPU_GPU
    Sampler() : type(Type::independent), ptr(nullptr) {}

    static Sampler *create_samplers(Type sampler_type, int samples_per_pixel, int size,
                                    GPUMemoryAllocator &allocator);

    Type get_type() const {
        return type;
    }

    PBRT_CPU_GPU
    explicit Sampler(IndependentSampler *independent_sampler) : type(Type::independent), ptr(independent_sampler) {}

    explicit Sampler(MLTSampler *mlt_sampler) : type(Type::mlt), ptr(mlt_sampler) {}

    PBRT_CPU_GPU
    explicit Sampler(StratifiedSampler *stratified_sampler) : type(Type::stratified), ptr(stratified_sampler) {}

    MLTSampler *get_mlt_sampler() const;

    PBRT_CPU_GPU
    void start_pixel_sample(int pixel_idx, int sample_idx, int dimension);

    PBRT_CPU_GPU
    void copy_state_from(const Sampler &source);

    PBRT_CPU_GPU
    int get_samples_per_pixel() const;

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
    void *ptr = nullptr;
};
