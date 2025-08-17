#include <pbrt/base/filter.h>
#include <pbrt/filters/filter_sampler.h>
#include <pbrt/filters/gaussian.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/util/math.h>

PBRT_CPU_GPU inline Real gaussian(Real x, Real mu = 0, Real sigma = 1) {
    return 1 / std::sqrt(2 * pbrt::PI * sigma * sigma) *
           std::exp(-sqr(x - mu) / (2 * sigma * sigma));
}

PBRT_CPU_GPU
inline Real gaussian_integral(Real x0, Real x1, Real mu = 0, Real sigma = 1) {
    Real sigmaRoot2 = sigma * Real(Sqrt2);
    return 0.5f * (std::erf((mu - x0) / sigmaRoot2) - std::erf((mu - x1) / sigmaRoot2));
}

GaussianFilter::GaussianFilter(const ParameterDictionary &parameters,
                               GPUMemoryAllocator &allocator) {
    auto xw = parameters.get_float("xradius", 1.5f);
    auto yw = parameters.get_float("yradius", 1.5f);

    radius = Vector2f(xw, yw);
    sigma = parameters.get_float("sigma", 0.5f);

    expX = gaussian(radius.x, 0, sigma);
    expY = gaussian(radius.y, 0, sigma);

    sampler = allocator.create<FilterSampler>(*this, allocator);
}

PBRT_CPU_GPU
Real GaussianFilter::evaluate(Point2f p) const {
    return (std::max<Real>(0, gaussian(p.x, 0, sigma) - expX) *
            std::max<Real>(0, gaussian(p.y, 0, sigma) - expY));
}

PBRT_CPU_GPU
Real GaussianFilter::get_integral() const {
    return (gaussian_integral(-radius.x, radius.x, 0, sigma) - 2 * radius.x * expX) *
           (gaussian_integral(-radius.y, radius.y, 0, sigma) - 2 * radius.y * expY);
}

PBRT_CPU_GPU
FilterSample GaussianFilter::sample(Point2f u) const {
    return sampler->sample(u);
}
