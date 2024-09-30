#pragma once

#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/util/macro.h"
#include <optional>
#include <string>
#include <vector>

class Film;
class Filter;
class GreyScaleFilm;
class IntegratorBase;
class MLTSampler;
class ParameterDictionary;
class Spectrum;

struct PathSample {
    FloatType x;
    FloatType y;
    // TODO: change x y to float like Film::AddSplat() from PBRT-V4

    SampledSpectrum radiance;
    SampledWavelengths lambda;

    PBRT_CPU_GPU
    PathSample(FloatType _x, FloatType _y, SampledSpectrum _radiance, SampledWavelengths _lambda)
        : x(_x), y(_y), radiance(_radiance), lambda(_lambda) {}
};

class MLTPathIntegrator {
  public:
    static MLTPathIntegrator *create(std::optional<int> samples_per_pixel,
                                     const ParameterDictionary &parameters,
                                     const IntegratorBase *base,
                                     std::vector<void *> &gpu_dynamic_pointers);

    void render(Film *film, GreyScaleFilm &heat_map, const Filter *filter);

    uint get_mutation_per_pixel() const {
        return mutation_per_pixel;
    }

    PBRT_CPU_GPU
    inline FloatType compute_luminance(const SampledSpectrum &radiance,
                                       const SampledWavelengths &lambda) const {
        return radiance.y(lambda, cie_y);
    }

    PBRT_GPU
    PathSample generate_new_path(Sampler *sampler, const Filter *filter) const;

    const IntegratorBase *base;
    Point2i film_dimension;

    Sampler *samplers;
    MLTSampler *mlt_samplers;
    uint mutation_per_pixel;

    uint max_depth;

    const Spectrum *cie_y;
};
