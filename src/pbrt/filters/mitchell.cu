#include "pbrt/base/filter.h"
#include "pbrt/filters/mitchell.h"

#include <pbrt/scene/parameter_dictionary.h>

MitchellFilter *MitchellFilter::create(const ParameterDictionary &parameters,
                                       std::vector<void *> &gpu_dynamic_pointers) {
    MitchellFilter *mitchell_filter;
    CHECK_CUDA_ERROR(cudaMallocManaged(&mitchell_filter, sizeof(MitchellFilter)));
    gpu_dynamic_pointers.push_back(mitchell_filter);

    auto xw = parameters.get_float("xradius", 2.f);
    auto yw = parameters.get_float("yradius", 2.f);
    auto b = parameters.get_float("B", 1.f / 3.f);
    auto c = parameters.get_float("C", 1.f / 3.f);

    mitchell_filter->init(Vector2f(xw, yw), b, c);

    return mitchell_filter;
}

void MitchellFilter::init_sampler(const Filter *filter, std::vector<void *> &gpu_dynamic_pointers) {
    sampler = FilterSampler::create(filter, gpu_dynamic_pointers);
}

PBRT_CPU_GPU
FloatType MitchellFilter::mitchell_1d(FloatType x) const {
    x = std::abs(x);
    if (x <= 1) {
        return ((12 - 9 * b - 6 * c) * x * x * x + (-18 + 12 * b + 6 * c) * x * x + (6 - 2 * b)) *
               (1.f / 6.f);
    }

    if (x <= 2) {
        return ((-b - 6 * c) * x * x * x + (6 * b + 30 * c) * x * x + (-12 * b - 48 * c) * x +
                (8 * b + 24 * c)) *
               (1.f / 6.f);
    }

    return 0;
}
