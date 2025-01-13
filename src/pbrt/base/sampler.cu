#include <pbrt/base/filter.h>
#include <pbrt/base/sampler.h>
#include <pbrt/samplers/independent.h>
#include <pbrt/samplers/mlt.h>
#include <pbrt/samplers/stratified.h>

static __global__ void init_independent_samplers(IndependentSampler *samplers,
                                                 uint samples_per_pixel, uint num) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num) {
        return;
    }

    samplers[idx].init(samples_per_pixel);
}

static __global__ void init_stratified_samplers(StratifiedSampler *samplers,
                                                uint samples_per_dimension, uint num) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num) {
        return;
    }

    samplers[idx].init(samples_per_dimension);
}

template <typename T>
static __global__ void init_samplers(Sampler *samplers, T *_samplers, uint length) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length) {
        return;
    }

    samplers[idx].init(&_samplers[idx]);
}

Sampler *Sampler::create_samplers_for_each_pixels(const std::string &sampler_type,
                                                  const uint samples_per_pixel,
                                                  const uint total_pixel_num,
                                                  GPUMemoryAllocator &allocator) {
    uint threads = 1024;
    uint blocks = divide_and_ceil(total_pixel_num, threads);

    auto samplers = allocator.allocate<Sampler>(total_pixel_num);

    if (sampler_type == "independent") {
        auto independent_samplers = allocator.allocate<IndependentSampler>(total_pixel_num);

        init_independent_samplers<<<blocks, threads>>>(independent_samplers, samples_per_pixel,
                                                       total_pixel_num);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        init_samplers<<<blocks, threads>>>(samplers, independent_samplers, total_pixel_num);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        return samplers;
    }

    if (sampler_type == "stratified") {
        if (sqr(std::sqrt(samples_per_pixel)) != samples_per_pixel) {
            REPORT_FATAL_ERROR();
        }

        auto samples_per_dimension = uint(std::sqrt(samples_per_pixel));
        // samples_per_pixel = samples_per_dimension * samples_per_dimension;

        auto stratified_samplers = allocator.allocate<StratifiedSampler>(total_pixel_num);

        init_stratified_samplers<<<blocks, threads>>>(stratified_samplers, samples_per_dimension,
                                                      total_pixel_num);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        init_samplers<<<blocks, threads>>>(samplers, stratified_samplers, total_pixel_num);

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
void Sampler::start_pixel_sample(uint pixel_idx, uint sample_idx, uint dimension) {
    switch (type) {
    case Type::independent: {
        static_cast<IndependentSampler *>(ptr)->start_pixel_sample(pixel_idx, sample_idx,
                                                                   dimension);
        return;
    }

    case Type::mlt: {
        static_cast<MLTSampler *>(ptr)->StartPixelSample(pixel_idx, sample_idx, dimension);

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
uint Sampler::get_samples_per_pixel() const {
    switch (type) {
    case Type::independent: {
        return static_cast<IndependentSampler *>(ptr)->get_samples_per_pixel();
    }

    case Type::mlt: {
        // TODO: implement get_samples_per_pixel() for MLTSampler?
        return 4;
    }

    case Type::stratified: {
        return static_cast<StratifiedSampler *>(ptr)->get_samples_per_pixel();
    }
    }

    REPORT_FATAL_ERROR();
    return 0;
}

PBRT_CPU_GPU
FloatType Sampler::get_1d() {
    switch (type) {
    case Type::independent: {
        return static_cast<IndependentSampler *>(ptr)->get_1d();
    }

    case Type::mlt: {
        return static_cast<MLTSampler *>(ptr)->Get1D();
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
        return static_cast<MLTSampler *>(ptr)->Get2D();
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
        return static_cast<MLTSampler *>(ptr)->GetPixel2D();
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
