#pragma once

#include <memory>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/spectrum.h>
#include <pbrt/gpu/macro.h>
#include <vector>

class Film;
class GPUMemoryAllocator;
class GreyScaleFilm;
class ParameterDictionary;
class Sampler;
struct PathSample;

class MLTPathIntegrator {
  public:
    static std::unique_ptr<MLTPathIntegrator> create(int mutations_per_pixel, const ParameterDictionary &parameters,
                                                     const IntegratorBase *base, const Film *film_prototype,
                                                     GPUMemoryAllocator &allocator);

    double render(Film *film, GreyScaleFilm &heat_map) const;

    Real compute_luminance(const SampledSpectrum &radiance, const SampledWavelengths &lambda) const {
        return radiance.y(lambda, cie_y.get());
    }

    Real compute_importance(const PathSample &path) const;

    PathSample generate_path_sample(Sampler *sampler) const;

    // The integrator state is CPU-owned. Its members still borrow the shared
    // scene objects created by SceneBuilder.
    std::unique_ptr<IntegratorBase> base;
    Point2i film_dimension;

    // The scene allocator owns these CUDA managed Films and their pixel buffers.
    std::vector<Film *> chain_films;

    int mutations_per_pixel;
    // More samplers give more, shallower Markov chains for the same sample budget.
    int num_samplers;
    long long mutations_per_sampler;
    Real sigma;
    Real large_step_probability;

    int max_depth;
    bool regularize;

    // Keep the Spectrum wrapper on the CPU while borrowing its sampled data.
    std::unique_ptr<Spectrum> cie_y;
};
