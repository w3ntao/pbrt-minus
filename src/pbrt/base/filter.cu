#include <pbrt/base/filter.h>
#include <pbrt/cameras/perspective.h>
#include <pbrt/filters/box.h>
#include <pbrt/filters/gaussian.h>
#include <pbrt/filters/mitchell.h>
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
    }

    REPORT_FATAL_ERROR();
    return Vector2f(NAN, NAN);
}

PBRT_CPU_GPU
FloatType Filter::get_integral() const {
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
    }

    REPORT_FATAL_ERROR();

    return NAN;
}

PBRT_CPU_GPU
FloatType Filter::evaluate(const Point2f p) const {
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
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_CPU_GPU
FilterSample Filter::sample(const Point2f u) const {
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
    }

    REPORT_FATAL_ERROR();
    return {};
}

const FilterSampler *FilterSampler::create(const Filter *filter, GPUMemoryAllocator &allocator) {
    auto filter_sampler = allocator.allocate<FilterSampler>();

    filter_sampler->init(filter, allocator);

    return filter_sampler;
}

void FilterSampler::init(const Filter *filter, GPUMemoryAllocator &allocator) {
    const auto filter_radius = filter->get_radius();

    domain = Bounds2f(Point2f(-filter_radius), Point2f(filter_radius));
    f.init(int(32 * filter_radius.x), int(32 * filter_radius.y), allocator);

    // Tabularize unnormalized filter function in _f_
    for (int y = 0; y < f.y_size(); ++y) {
        for (int x = 0; x < f.x_size(); ++x) {
            Point2f p = domain.lerp(Point2f((x + 0.5f) / f.x_size(), (y + 0.5f) / f.y_size()));
            f(x, y) = filter->evaluate(p);
        }
    }

    distrib.init(&f, domain, allocator);
}
