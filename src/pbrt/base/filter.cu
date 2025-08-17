#include <pbrt/base/filter.h>
#include <pbrt/filters/filter_sampler.h>
#include <pbrt/gpu/gpu_memory_allocator.h>

const Filter *Filter::create(const std::string &filter_type, const ParameterDictionary &parameters,
                             GPUMemoryAllocator &allocator) {
    if (filter_type == "box") {
        return allocator.create<Filter>(BoxFilter(parameters));
    }

    if (filter_type == "gaussian") {
        return allocator.create<Filter>(GaussianFilter(parameters, allocator));
    }

    if (filter_type == "mitchell") {
        return allocator.create<Filter>(MitchellFilter(parameters, allocator));
    }

    if (filter_type == "triangle") {
        return allocator.create<Filter>(TriangleFilter(parameters));
    }

    REPORT_FATAL_ERROR();
    return nullptr;
}

PBRT_CPU_GPU
FilterSample Filter::sample(const Point2f &u) const {
    return cuda::std::visit([&](auto &x) { return x.sample(u); }, *this);
}
