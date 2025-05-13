#pragma once

#include <pbrt/gpu/macro.h>
#include <pbrt/samplers/mlt.h>
#include <pbrt/spectrum_util/sampled_spectrum.h>

class Film;
class GreyScaleFilm;
class GPUMemoryAllocator;
class MLTSampler;
class ParameterDictionary;
class Spectrum;
struct IntegratorBase;

struct PathSample {
    Point2f p_film;
    SampledSpectrum radiance;
    SampledWavelengths lambda;

    PBRT_CPU_GPU
    PathSample(const Point2f &_p_film, const SampledSpectrum &_radiance,
               const SampledWavelengths &_lambda)
        : p_film(_p_film), radiance(_radiance), lambda(_lambda) {}
};

class MLTPathIntegrator {
  public:
    static MLTPathIntegrator *create(int mutations_per_pixel, const ParameterDictionary &parameters,
                                     const IntegratorBase *base, GPUMemoryAllocator &allocator);

    double render(Film *film, GreyScaleFilm &heat_map, uint mutations_per_pixel, bool preview);

    PBRT_CPU_GPU
    Real compute_luminance(const SampledSpectrum &radiance,
                                const SampledWavelengths &lambda) const {
        return radiance.y(lambda, cie_y);
    }

    PBRT_CPU_GPU
    PathSample generate_path_sample(Sampler *sampler) const;

    const IntegratorBase *base;
    Point2i film_dimension;

    Sampler *samplers;
    MLTSampler *mlt_samplers;

    uint max_depth;
    bool regularize;

    const Spectrum *cie_y;
};
