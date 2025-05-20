#include <pbrt/base/filter.h>
#include <pbrt/filters/filter_sampler.h>
#include <pbrt/gpu/gpu_memory_allocator.h>

const Filter *Filter::create(const std::string &filter_type, const ParameterDictionary &parameters,
                             GPUMemoryAllocator &allocator) {
    auto filter = allocator.allocate<Filter>();

    if (filter_type == "box") {
        *filter = BoxFilter(parameters);
        return filter;
    }

    if (filter_type == "gaussian") {
        *filter = GaussianFilter(parameters, allocator);
        return filter;
    }

    if (filter_type == "mitchell") {
        *filter = MitchellFilter(parameters, allocator);
        return filter;
    }

    if (filter_type == "triangle") {
        *filter = TriangleFilter(parameters);
        return filter;
    }

    REPORT_FATAL_ERROR();
    return nullptr;
}

PBRT_CPU_GPU
FilterSample Filter::sample(const Point2f &u) const {
    return cuda::std::visit([&](auto &x) { return x.sample(u); }, *this);
}
