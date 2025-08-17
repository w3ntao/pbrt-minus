#include <pbrt/base/filter.h>
#include <pbrt/filters/filter_sampler.h>

FilterSampler::FilterSampler(const Filter &filter, GPUMemoryAllocator &allocator) {
    const auto filter_radius = filter.get_radius();

    domain = Bounds2f(Point2f(-filter_radius), Point2f(filter_radius));
    f = Array2D<Real>(static_cast<int>(32 * filter_radius.x),
                      static_cast<int>(32 * filter_radius.y), allocator);

    // Tabularize unnormalized filter function in _f_
    for (int y = 0; y < f.y_size(); ++y) {
        for (int x = 0; x < f.x_size(); ++x) {
            Point2f p = domain.lerp(Point2f((x + 0.5f) / f.x_size(), (y + 0.5f) / f.y_size()));
            f(x, y) = filter.evaluate(p);
        }
    }

    distrib = PiecewiseConstant2D(&f, domain, allocator);
}
