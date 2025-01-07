#pragma once

#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/util/macro.h"
#include <optional>
#include <vector>

class Film;
class GreyScaleFilm;
class MLTSampler;
class ParameterDictionary;
class Spectrum;
struct IntegratorBase;

struct PathSample {
    Point2f p_film;
    SampledSpectrum radiance;
    SampledWavelengths lambda;

    PBRT_CPU_GPU
    PathSample(const Point2f _p_film, SampledSpectrum _radiance, SampledWavelengths _lambda)
        : p_film(_p_film), radiance(_radiance), lambda(_lambda) {}
};

class MLTPathIntegrator {
  public:
    static MLTPathIntegrator *create(const ParameterDictionary &parameters,
                                     const IntegratorBase *base,
                                     std::vector<void *> &gpu_dynamic_pointers);

    void render(Film *film, GreyScaleFilm &heat_map, uint mutations_per_pixel, bool preview);

    PBRT_CPU_GPU
    FloatType compute_luminance(const SampledSpectrum &radiance,
                                const SampledWavelengths &lambda) const {
        return radiance.y(lambda, cie_y);
    }

    PBRT_GPU
    PathSample generate_path_sample(Sampler *sampler) const;

    const IntegratorBase *base;
    Point2i film_dimension;

    Sampler *samplers;
    MLTSampler *mlt_samplers;

    uint max_depth;
    bool regularize;

    const Spectrum *cie_y;
};
