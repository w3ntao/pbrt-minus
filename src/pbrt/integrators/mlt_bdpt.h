#pragma once

#include <pbrt/base/spectrum.h>

class Film;
class GPUMemoryAllocator;
class GreyScaleFilm;
class ParameterDictionary;

class MLTSampler;
class Sampler;

struct BDPTConfig;
struct FilmSample;
struct IntegratorBase;
struct Vertex;

struct BDPTPathSample {
    Point2f p_film;
    SampledSpectrum radiance;
    SampledWavelengths lambda;

    PBRT_CPU_GPU
    static BDPTPathSample zero() {
        return BDPTPathSample(Point2f(NAN, NAN), 0, SampledWavelengths::zero());
    }

    PBRT_CPU_GPU
    BDPTPathSample(const Point2f &_p_film, const SampledSpectrum &_radiance,
                   const SampledWavelengths &_lambda)
        : p_film(_p_film), radiance(_radiance), lambda(_lambda) {}
};

class MLTBDPTIntegrator {
  public:
    static constexpr int cameraStreamIndex = 0;
    static constexpr int lightStreamIndex = 1;
    static constexpr int connectionStreamIndex = 2;
    static constexpr int nSampleStreams = 3;

    static MLTBDPTIntegrator *create(int mutations_per_pixel, const ParameterDictionary &parameters,
                                     const IntegratorBase *base, GPUMemoryAllocator &allocator);

    double render(Film *film, GreyScaleFilm &heat_map, uint mutations_per_pixel, bool preview);

    PBRT_GPU
    BDPTPathSample li(int depth, Sampler *sampler, Vertex *camera_vertices,
                      Vertex *light_vertices) const;

    PBRT_CPU_GPU
    Real compute_luminance(const SampledSpectrum &radiance,
                           const SampledWavelengths &lambda) const {
        return radiance.y(lambda, cie_y);
    }

    Sampler *samplers;
    MLTSampler *mlt_samplers;

    const BDPTConfig *config;

  private:
    Point2i film_dimension;

    const Spectrum *cie_y;
};
