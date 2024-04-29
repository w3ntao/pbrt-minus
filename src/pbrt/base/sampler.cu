#include "pbrt/base/sampler.h"
#include "pbrt/base/filter.h"
#include "pbrt/samplers/independent_sampler.h"

PBRT_CPU_GPU
void Sampler::init(IndependentSampler *independent_sampler) {
    sampler_type = SamplerType::independent_sampler;
    sampler_ptr = independent_sampler;
}

PBRT_GPU
void Sampler::start_pixel_sample(uint pixel_idx, uint sample_idx, uint dimension) {
    switch (sampler_type) {
    case (SamplerType::independent_sampler): {
        ((IndependentSampler *)sampler_ptr)->start_pixel_sample(pixel_idx, sample_idx, dimension);
        return;
    }
    }
    report_function_error_and_exit(__func__);
}

PBRT_CPU_GPU
uint Sampler::get_samples_per_pixel() const {
    switch (sampler_type) {
    case (SamplerType::independent_sampler): {
        return ((IndependentSampler *)sampler_ptr)->get_samples_per_pixel();
    }
    }

    report_function_error_and_exit(__func__);
    return 0;
}

PBRT_GPU
FloatType Sampler::get_1d() {
    switch (sampler_type) {
    case (SamplerType::independent_sampler): {
        return ((IndependentSampler *)sampler_ptr)->get_1d();
    }
    }

    report_function_error_and_exit(__func__);
    return NAN;
}

PBRT_GPU
Point2f Sampler::get_2d() {
    switch (sampler_type) {
    case (SamplerType::independent_sampler): {
        return ((IndependentSampler *)sampler_ptr)->get_2d();
    }
    }

    report_function_error_and_exit(__func__);
}

PBRT_GPU
Point2f Sampler::get_pixel_2d() {
    switch (sampler_type) {
    case (SamplerType::independent_sampler): {
        return ((IndependentSampler *)sampler_ptr)->get_pixel_2d();
    }
    }

    report_function_error_and_exit(__func__);
}

PBRT_GPU
CameraSample Sampler::get_camera_sample(const Point2i pPixel, const Filter *filter) {
    auto fs = filter->sample(get_pixel_2d());

    return CameraSample(pPixel.to_point2f() + fs.p + Vector2f(0.5, 0.5), get_2d(), fs.weight);
}
