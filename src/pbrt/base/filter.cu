#include <pbrt/base/filter.h>
#include <pbrt/cameras/perspective.h>
#include <pbrt/filters/box.h>
#include <pbrt/filters/filter_sampler.h>
#include <pbrt/filters/gaussian.h>
#include <pbrt/filters/mitchell.h>
#include <pbrt/filters/triangle.h>
#include <pbrt/gpu/gpu_memory_allocator.h>

const Filter *Filter::create(const std::string &filter_type, const ParameterDictionary &parameters,
                             GPUMemoryAllocator &allocator) {
    auto filter = allocator.allocate<Filter>();

    if (filter_type == "box") {
        auto box_filter = BoxFilter::create(parameters, allocator);
        filter->init(box_filter);

        return filter;
    }

    if (filter_type == "gaussian") {
        auto gaussian_filter = GaussianFilter::create(parameters, allocator);
        filter->init(gaussian_filter);
        gaussian_filter->init_sampler(filter, allocator);

        return filter;
    }

    if (filter_type == "mitchell") {
        auto mitchell_filter = MitchellFilter::create(parameters, allocator);
        filter->init(mitchell_filter);
        mitchell_filter->init_sampler(filter, allocator);

        return filter;
    }

    if (filter_type == "triangle") {
        auto triangle_filter = TriangleFilter::create(parameters, allocator);
        filter->init(triangle_filter);

        return filter;
    }

    REPORT_FATAL_ERROR();
    return nullptr;
}

void Filter::init(const BoxFilter *box_filter) {
    ptr = box_filter;
    type = Type::box;
}

void Filter::init(const GaussianFilter *gaussian_filter) {
    ptr = gaussian_filter;
    type = Type::gaussian;
}

void Filter::init(const MitchellFilter *mitchell_filter) {
    ptr = mitchell_filter;
    type = Type::mitchell;
}

void Filter::init(const TriangleFilter *triangle_filter) {
    ptr = triangle_filter;
    type = Type::triangle;
}

PBRT_CPU_GPU
Vector2f Filter::get_radius() const {
    switch (type) {
    case Type::box: {
        return static_cast<const BoxFilter *>(ptr)->get_radius();
    }

    case Type::gaussian: {
        return static_cast<const GaussianFilter *>(ptr)->get_radius();
    }

    case Type::mitchell: {
        return static_cast<const MitchellFilter *>(ptr)->get_radius();
    }

    case Type::triangle: {
        return static_cast<const TriangleFilter *>(ptr)->get_radius();
    }
    }

    REPORT_FATAL_ERROR();
    return Vector2f(NAN, NAN);
}

PBRT_CPU_GPU
Real Filter::get_integral() const {
    switch (type) {
    case Type::box: {
        return static_cast<const BoxFilter *>(ptr)->get_integral();
    }

    case Type::gaussian: {
        return static_cast<const GaussianFilter *>(ptr)->get_integral();
    }

    case Type::mitchell: {
        return static_cast<const MitchellFilter *>(ptr)->get_integral();
    }

    case Type::triangle: {
        return static_cast<const TriangleFilter *>(ptr)->get_integral();
    }
    }

    REPORT_FATAL_ERROR();

    return NAN;
}

PBRT_CPU_GPU
Real Filter::evaluate(const Point2f &p) const {
    switch (type) {
    case Type::box: {
        return static_cast<const BoxFilter *>(ptr)->evaluate(p);
    }

    case Type::gaussian: {
        return static_cast<const GaussianFilter *>(ptr)->evaluate(p);
    }

    case Type::mitchell: {
        return static_cast<const MitchellFilter *>(ptr)->evaluate(p);
    }

    case Type::triangle: {
        return static_cast<const TriangleFilter *>(ptr)->evaluate(p);
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_CPU_GPU
FilterSample Filter::sample(const Point2f &u) const {
    switch (type) {
    case Type::box: {
        return static_cast<const BoxFilter *>(ptr)->sample(u);
    }

    case Type::gaussian: {
        return static_cast<const GaussianFilter *>(ptr)->sample(u);
    }

    case Type::mitchell: {
        return static_cast<const MitchellFilter *>(ptr)->sample(u);
    }

    case Type::triangle: {
        return static_cast<const TriangleFilter *>(ptr)->sample(u);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}
