#include "pbrt/base/filter.h"
#include "pbrt/filters/mitchell.h"

void MitchellFilter::build_filter_sampler(const Filter *filter,
                                          std::vector<void *> &gpu_dynamic_pointers) {
    sampler = FilterSampler::create(filter, gpu_dynamic_pointers);
}

PBRT_CPU_GPU
FloatType MitchellFilter::Mitchell1D(FloatType x) const {
    x = std::abs(x);
    if (x <= 1) {
        return ((12 - 9 * b - 6 * c) * x * x * x + (-18 + 12 * b + 6 * c) * x * x + (6 - 2 * b)) *
               (1.f / 6.f);
    } else if (x <= 2) {
        return ((-b - 6 * c) * x * x * x + (6 * b + 30 * c) * x * x + (-12 * b - 48 * c) * x +
                (8 * b + 24 * c)) *
               (1.f / 6.f);
    }

    return 0;
}
