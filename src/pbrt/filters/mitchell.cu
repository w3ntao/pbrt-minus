#include <pbrt/base/filter.h>
#include <pbrt/filters/filter_sampler.h>
#include <pbrt/filters/mitchell.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>

MitchellFilter::MitchellFilter(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator)
    : radius(NAN, NAN), sampler(nullptr) {
    auto xw = parameters.get_float("xradius", 2.f);
    auto yw = parameters.get_float("yradius", 2.f);
    radius = Vector2f(xw, yw);

    b = parameters.get_float("B", 1.f / 3.f);
    c = parameters.get_float("C", 1.f / 3.f);

    sampler = allocator.create<FilterSampler>(*this, allocator);
}

PBRT_CPU_GPU
FilterSample MitchellFilter::sample(const Point2f u) const {
    return sampler->sample(u);
}

PBRT_CPU_GPU
Real MitchellFilter::mitchell_1d(Real x) const {
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
