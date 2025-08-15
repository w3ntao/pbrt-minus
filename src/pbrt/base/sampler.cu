#include <pbrt/base/filter.h>
#include <pbrt/base/sampler.h>
#include <pbrt/cameras/camera_base.h>
#include <pbrt/filters/filter_sampler.h>
#include <pbrt/samplers/independent.h>
#include <pbrt/samplers/mlt.h>
#include <pbrt/samplers/stratified.h>

template <typename TypeOfSampler>
static __global__ void init_samplers(Sampler *samplers, TypeOfSampler *_samplers,
                                     int samples_per_pixel, int size) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) {
        return;
    }
    _samplers[idx] = TypeOfSampler(samples_per_pixel);
    samplers[idx].init(&_samplers[idx]);
}

Sampler *Sampler::create_samplers(const std::string &string_sampler_type,
                                  const int samples_per_pixel, const int size,
                                  GPUMemoryAllocator &allocator) {
    const int blocks = divide_and_ceil(size, MAX_THREADS_PER_BLOCKS);

    auto samplers = allocator.allocate<Sampler>(size);

    auto sampler_type = parse_sampler_type(string_sampler_type);

    if (sampler_type == Type::independent) {
        auto independent_samplers = allocator.allocate<IndependentSampler>(size);
        init_samplers<<<blocks, MAX_THREADS_PER_BLOCKS>>>(samplers, independent_samplers,
                                                          samples_per_pixel, size);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        return samplers;
    }

    if (sampler_type == Type::stratified) {
        auto stratified_samplers = allocator.allocate<StratifiedSampler>(size);
        init_samplers<<<blocks, MAX_THREADS_PER_BLOCKS>>>(samplers, stratified_samplers,
                                                          samples_per_pixel, size);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        return samplers;
    }

    REPORT_FATAL_ERROR();
    return nullptr;
}

PBRT_CPU_GPU
void Sampler::init(IndependentSampler *independent_sampler) {
    type = Type::independent;
    ptr = independent_sampler;
}

PBRT_CPU_GPU
void Sampler::init(MLTSampler *mlt_sampler) {
    type = Type::mlt;
    ptr = mlt_sampler;
}

PBRT_CPU_GPU
void Sampler::init(StratifiedSampler *stratified_sampler) {
    type = Type::stratified;
    ptr = stratified_sampler;
}

PBRT_CPU_GPU
MLTSampler *Sampler::get_mlt_sampler() const {
    if (type != Type::mlt) {
        REPORT_FATAL_ERROR();
    }

    return static_cast<MLTSampler *>(ptr);
}

PBRT_CPU_GPU
void Sampler::start_pixel_sample(int pixel_idx, int sample_idx, int dimension) {
    switch (type) {
    case Type::independent: {
        static_cast<IndependentSampler *>(ptr)->start_pixel_sample(pixel_idx, sample_idx,
                                                                   dimension);
        return;
    }

    case Type::mlt: {
        printf("\nERROR: you should never invoke %s() for MLT sampler\n", __func__);
        REPORT_FATAL_ERROR();
        return;
    }

    case Type::stratified: {
        static_cast<StratifiedSampler *>(ptr)->start_pixel_sample(pixel_idx, sample_idx, dimension);
        return;
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
int Sampler::get_samples_per_pixel() const {
    switch (type) {
    case Type::independent: {
        return static_cast<const IndependentSampler *>(ptr)->get_samples_per_pixel();
    }

    case Type::mlt: {
        return static_cast<const MLTSampler *>(ptr)->get_samples_per_pixel();
    }

    case Type::stratified: {
        return static_cast<const StratifiedSampler *>(ptr)->get_samples_per_pixel();
    }
    }

    REPORT_FATAL_ERROR();
    return 0;
}

PBRT_CPU_GPU
Real Sampler::get_1d() {
    switch (type) {
    case Type::independent: {
        return static_cast<IndependentSampler *>(ptr)->get_1d();
    }

    case Type::mlt: {
        return static_cast<MLTSampler *>(ptr)->get_1d();
    }

    case Type::stratified: {
        return static_cast<StratifiedSampler *>(ptr)->get_1d();
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_CPU_GPU
Point2f Sampler::get_2d() {
    switch (type) {
    case Type::independent: {
        return static_cast<IndependentSampler *>(ptr)->get_2d();
    }

    case Type::mlt: {
        return static_cast<MLTSampler *>(ptr)->get_2d();
    }

    case Type::stratified: {
        return static_cast<StratifiedSampler *>(ptr)->get_2d();
    }
    }

    REPORT_FATAL_ERROR();
    return Point2f(NAN, NAN);
}

PBRT_CPU_GPU
Point2f Sampler::get_pixel_2d() {
    switch (type) {
    case Type::independent: {
        return static_cast<IndependentSampler *>(ptr)->get_pixel_2d();
    }

    case Type::mlt: {
        return static_cast<MLTSampler *>(ptr)->get_pixel_2d();
    }

    case Type::stratified: {
        return static_cast<StratifiedSampler *>(ptr)->get_pixel_2d();
    }
    }

    REPORT_FATAL_ERROR();
    return Point2f(NAN, NAN);
}

PBRT_CPU_GPU
CameraSample Sampler::get_camera_sample(const Point2i pPixel, const Filter *filter) {
    auto fs = filter->sample(get_pixel_2d());

    return CameraSample(pPixel.to_point2f() + fs.p + Vector2f(0.5, 0.5), fs.weight);
}
