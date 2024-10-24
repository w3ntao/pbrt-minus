#include "pbrt/base/filter.h"
#include "pbrt/filters/box.h"
#include "pbrt/filters/mitchell.h"

const Filter *Filter::create_box_filter(FloatType radius,
                                        std::vector<void *> &gpu_dynamic_pointers) {
    BoxFilter *box_filter;
    CHECK_CUDA_ERROR(cudaMallocManaged(&box_filter, sizeof(BoxFilter)));

    Filter *filter;
    CHECK_CUDA_ERROR(cudaMallocManaged(&filter, sizeof(Filter)));

    gpu_dynamic_pointers.push_back(box_filter);
    gpu_dynamic_pointers.push_back(filter);

    box_filter->init(radius);
    filter->init(box_filter);

    return filter;
}

const Filter *Filter::create_mitchell_filter(const Vector2f &radius, FloatType b, FloatType c,
                                             std::vector<void *> &gpu_dynamic_pointers) {
    MitchellFilter *mitchell_filter;
    CHECK_CUDA_ERROR(cudaMallocManaged(&mitchell_filter, sizeof(MitchellFilter)));
    gpu_dynamic_pointers.push_back(mitchell_filter);
    mitchell_filter->init(radius, b, c);

    Filter *filter;
    CHECK_CUDA_ERROR(cudaMallocManaged(&filter, sizeof(Filter)));
    gpu_dynamic_pointers.push_back(filter);

    filter->init(mitchell_filter);
    mitchell_filter->build_filter_sampler(filter, gpu_dynamic_pointers);

    return filter;
}

void Filter::init(const BoxFilter *box_filter) {
    ptr = box_filter;
    type = Type::box;
}

void Filter::init(const MitchellFilter *mitchell_filter) {
    ptr = mitchell_filter;
    type = Type::mitchell;
}

PBRT_CPU_GPU
Vector2f Filter::radius() const {
    switch (type) {
    case (Type::box): {
        return static_cast<const BoxFilter *>(ptr)->get_radius();
    }

    case (Type::mitchell): {
        return static_cast<const MitchellFilter *>(ptr)->get_radius();
    }
    }

    REPORT_FATAL_ERROR();
    return Vector2f(NAN, NAN);
}

PBRT_CPU_GPU
FloatType Filter::evaluate(const Point2f p) const {
    switch (type) {
    case (Type::mitchell): {
        return static_cast<const MitchellFilter *>(ptr)->evaluate(p);
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_CPU_GPU
FilterSample Filter::sample(const Point2f u) const {
    switch (type) {
    case (Type::box): {
        return static_cast<const BoxFilter *>(ptr)->sample(u);
    }

    case (Type::mitchell): {
        return static_cast<const MitchellFilter *>(ptr)->sample(u);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

const FilterSampler *FilterSampler::create(const Filter *filter,
                                           std::vector<void *> &gpu_dynamic_pointers) {
    FilterSampler *filter_sampler;
    cudaMallocManaged(&filter_sampler, sizeof(FilterSampler));
    gpu_dynamic_pointers.push_back(filter_sampler);

    filter_sampler->init(filter, gpu_dynamic_pointers);

    return filter_sampler;
}

void FilterSampler::init(const Filter *filter, std::vector<void *> &gpu_dynamic_pointers) {
    const auto filter_radius = filter->radius();

    domain = Bounds2f(Point2f(-filter_radius), Point2f(filter_radius));
    f.init(int(32 * filter_radius.x), int(32 * filter_radius.y), gpu_dynamic_pointers);

    // Tabularize unnormalized filter function in _f_
    for (int y = 0; y < f.y_size(); ++y) {
        for (int x = 0; x < f.x_size(); ++x) {
            Point2f p = domain.lerp(Point2f((x + 0.5f) / f.x_size(), (y + 0.5f) / f.y_size()));
            f(x, y) = filter->evaluate(p);
        }
    }

    distrib.init(&f, domain, gpu_dynamic_pointers);
}
