#pragma once

#include <pbrt/base/spectrum.h>
#include <pbrt/gpu/macro.h>
#include <pbrt/samplers/mlt.h>

class Film;
class GreyScaleFilm;
class GPUMemoryAllocator;
class MLTSampler;
class ParameterDictionary;
struct IntegratorBase;
struct PathSample;

class MLTPathIntegrator {

  public:
    static MLTPathIntegrator *create(int mutations_per_pixel, const ParameterDictionary &parameters,
                                     const IntegratorBase *base, GPUMemoryAllocator &allocator);

    double render(Film *film, GreyScaleFilm &heat_map, int mutations_per_pixel, bool preview);

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

    int max_depth;
    bool regularize;

    const Spectrum *cie_y;
};
