#include "pbrt/base/sampler.h"
#include "pbrt/base/filter.h"
#include "pbrt/samplers/independent_sampler.h"

PBRT_CPU_GPU
void Sampler::init(IndependentSampler *independent_sampler) {
    type = Type::independent_sampler;
    ptr = independent_sampler;
}

PBRT_GPU
void Sampler::start_pixel_sample(uint pixel_idx, uint sample_idx, uint dimension) {
    switch (type) {
    case (Type::independent_sampler): {
        ((IndependentSampler *)ptr)->start_pixel_sample(pixel_idx, sample_idx, dimension);
        return;
    }
    }
    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
uint Sampler::get_samples_per_pixel() const {
    switch (type) {
    case (Type::independent_sampler): {
        return ((IndependentSampler *)ptr)->get_samples_per_pixel();
    }
    }

    REPORT_FATAL_ERROR();
    return 0;
}

PBRT_GPU
FloatType Sampler::get_1d() {
    switch (type) {
    case (Type::independent_sampler): {
        return ((IndependentSampler *)ptr)->get_1d();
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_GPU
Point2f Sampler::get_2d() {
    switch (type) {
    case (Type::independent_sampler): {
        return ((IndependentSampler *)ptr)->get_2d();
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_GPU
Point2f Sampler::get_pixel_2d() {
    switch (type) {
    case (Type::independent_sampler): {
        return ((IndependentSampler *)ptr)->get_pixel_2d();
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_GPU
CameraSample Sampler::get_camera_sample(const Point2i pPixel, const Filter *filter) {
    auto fs = filter->sample(get_pixel_2d());

    return CameraSample(pPixel.to_point2f() + fs.p + Vector2f(0.5, 0.5), get_2d(), fs.weight);
}
