#include <pbrt/base/filter.h>
#include <pbrt/filters/filter_sampler.h>

const FilterSampler *FilterSampler::create(const Filter &filter, GPUMemoryAllocator &allocator) {
    auto filter_sampler = allocator.allocate<FilterSampler>();

    filter_sampler->init(filter, allocator);

    return filter_sampler;
}

void FilterSampler::init(const Filter &filter, GPUMemoryAllocator &allocator) {
    const auto filter_radius = filter.get_radius();

    domain = Bounds2f(Point2f(-filter_radius), Point2f(filter_radius));
    f.init(int(32 * filter_radius.x), int(32 * filter_radius.y), allocator);

    // Tabularize unnormalized filter function in _f_
    for (int y = 0; y < f.y_size(); ++y) {
        for (int x = 0; x < f.x_size(); ++x) {
            Point2f p = domain.lerp(Point2f((x + 0.5f) / f.x_size(), (y + 0.5f) / f.y_size()));
            f(x, y) = filter.evaluate(p);
        }
    }

    distrib.init(&f, domain, allocator);
}
