#include "pbrt/base/filter.h"
#include "pbrt/base/sampler.h"
#include "pbrt/samplers/independent_sampler.h"
#include "pbrt/samplers/stratified_sampler.h"

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

Sampler *Sampler::create(const std::string &sampler_type, const uint samples_per_pixel,
                         const uint total_pixel_num, std::vector<void *> &gpu_dynamic_pointers) {
    uint threads = 1024;
    uint blocks = divide_and_ceil(total_pixel_num, threads);

    if (sampler_type == "independent") {
        Sampler *samplers;
        IndependentSampler *independent_samplers;
        CHECK_CUDA_ERROR(cudaMallocManaged(&samplers, sizeof(Sampler) * total_pixel_num));
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&independent_samplers, sizeof(IndependentSampler) * total_pixel_num));

        gpu_dynamic_pointers.push_back(samplers);
        gpu_dynamic_pointers.push_back(independent_samplers);

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

        Sampler *samplers;
        StratifiedSampler *stratified_samplers;
        CHECK_CUDA_ERROR(cudaMallocManaged(&samplers, sizeof(Sampler) * total_pixel_num));
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&stratified_samplers, sizeof(StratifiedSampler) * total_pixel_num));
        gpu_dynamic_pointers.push_back(samplers);
        gpu_dynamic_pointers.push_back(stratified_samplers);

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

PBRT_CPU_GPU void Sampler::init(IndependentSampler *independent_sampler) {
    type = Type::independent;
    ptr = independent_sampler;
}

PBRT_CPU_GPU
void Sampler::init(StratifiedSampler *stratified_sampler) {
    type = Type::stratified;
    ptr = stratified_sampler;
}

PBRT_GPU
void Sampler::start_pixel_sample(uint pixel_idx, uint sample_idx, uint dimension) {
    switch (type) {
    case (Type::independent): {
        ((IndependentSampler *)ptr)->start_pixel_sample(pixel_idx, sample_idx, dimension);
        return;
    }

    case (Type::stratified): {
        ((StratifiedSampler *)ptr)->start_pixel_sample(pixel_idx, sample_idx, dimension);
        return;
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
uint Sampler::get_samples_per_pixel() const {
    switch (type) {
    case (Type::independent): {
        return ((IndependentSampler *)ptr)->get_samples_per_pixel();
    }

    case (Type::stratified): {
        return ((StratifiedSampler *)ptr)->get_samples_per_pixel();
    }
    }

    REPORT_FATAL_ERROR();
    return 0;
}

PBRT_GPU
FloatType Sampler::get_1d() {
    switch (type) {
    case (Type::independent): {
        return ((IndependentSampler *)ptr)->get_1d();
    }

    case (Type::stratified): {
        return ((StratifiedSampler *)ptr)->get_1d();
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_GPU
Point2f Sampler::get_2d() {
    switch (type) {
    case (Type::independent): {
        return ((IndependentSampler *)ptr)->get_2d();
    }

    case (Type::stratified): {
        return ((StratifiedSampler *)ptr)->get_2d();
    }
    }

    REPORT_FATAL_ERROR();
    return Point2f(NAN, NAN);
}

PBRT_GPU
Point2f Sampler::get_pixel_2d() {
    switch (type) {
    case (Type::independent): {
        return ((IndependentSampler *)ptr)->get_pixel_2d();
    }

    case (Type::stratified): {
        return ((StratifiedSampler *)ptr)->get_pixel_2d();
    }
    }

    REPORT_FATAL_ERROR();
    return Point2f(NAN, NAN);
}

PBRT_GPU
CameraSample Sampler::get_camera_sample(const Point2i pPixel, const Filter *filter) {
    auto fs = filter->sample(get_pixel_2d());

    return CameraSample(pPixel.to_point2f() + fs.p + Vector2f(0.5, 0.5), fs.weight);
}
