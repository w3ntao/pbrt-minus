#include <pbrt/filters/gaussian.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/util/math.h>

#include <pbrt/gpu/gpu_memory_allocator.h>

PBRT_CPU_GPU inline Real gaussian(Real x, Real mu = 0, Real sigma = 1) {
    return 1 / std::sqrt(2 * compute_pi() * sigma * sigma) *
           std::exp(-sqr(x - mu) / (2 * sigma * sigma));
}

PBRT_CPU_GPU
inline Real gaussian_integral(Real x0, Real x1, Real mu = 0,
                                   Real sigma = 1) {
    Real sigmaRoot2 = sigma * Real(Sqrt2);
    return 0.5f * (std::erf((mu - x0) / sigmaRoot2) - std::erf((mu - x1) / sigmaRoot2));
}

GaussianFilter *GaussianFilter::create(const ParameterDictionary &parameters,
                                       GPUMemoryAllocator &allocator) {
    auto xw = parameters.get_float("xradius", 1.5f);
    auto yw = parameters.get_float("yradius", 1.5f);
    auto sigma = parameters.get_float("sigma", 0.5f);

    auto gaussian_filter = allocator.allocate<GaussianFilter>();

    gaussian_filter->init(Vector2f(xw, yw), sigma);

    return gaussian_filter;
}

void GaussianFilter::init_sampler(const Filter *filter, GPUMemoryAllocator &allocator) {
    sampler = FilterSampler::create(filter, allocator);
}

void GaussianFilter::init(const Vector2f &_radius, Real _sigma) {
    radius = _radius;
    sigma = _sigma;

    expX = gaussian(radius.x, 0, sigma);
    expY = gaussian(radius.y, 0, sigma);

    sampler = nullptr;
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
