#pragma once

#include <pbrt/base/ray.h>
#include <pbrt/base/sampler.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/sampled_spectrum.h>

class Film;
class GreyScaleFilm;
class GPUMemoryAllocator;
class Spectrum;
struct IntegratorBase;

struct PTSample {
    SampledSpectrum radiance;
    Real depth = 0;
};

class MegakernelPathIntegrator {
  public:
    const int RAY_SPLIT_DEPTH;
    // 0: no split, otherwise split for the first n bounces (for transmissive dielectric materials only)

    struct PathContinuation;

    MegakernelPathIntegrator(int samples_per_pixel, Sampler::Type sampler_type, const ParameterDictionary &parameters,
                             const IntegratorBase *integrator_base, GPUMemoryAllocator &allocator, int ray_split_depth,
                             bool use_russian_roulette);

    void render(Film *film, GreyScaleFilm &firefly_mask, const std::string &output_filename, bool preview) const;

    // Renders independent camera paths that only update transformed Film
    // statistics and mean path depth for each pixel. Zero reconstruction
    // weight leaves image pixels untouched; path_depths must be GPU-accessible
    // and hold one value per pixel.
    void render_denoising_statistics(Film *film, Real *path_depths, bool match_bdpt_depth = false) const;

    PBRT_CPU_GPU
    // Selective PT can match BDPT's scattering-depth limit while legacy callers
    // retain their existing truncation and emissive-surface behavior.
    // A dielectric hit at depth zero recursively traces reflection and transmission.
    // The reported depth is the maximum depth of the resulting paths.
    static PTSample evaluate_Li_volume(const Ray &primary_ray, SampledWavelengths &lambda, Sampler *sampler,
                                       const IntegratorBase *base, int max_depth, bool regularize,
                                       bool match_bdpt_depth = false, Real path_depth = 0.0,
                                       const PathContinuation *continuation = nullptr, int ray_split_depth = 2,
                                       bool use_russian_roulette = true);

    PBRT_CPU_GPU
    static SampledSpectrum sample_Ld_volume(const SurfaceInteraction &surface_interaction, const BSDF *bsdf,
                                            const SampledWavelengths &lambda, Sampler *sampler,
                                            const IntegratorBase *base);

    PBRT_CPU_GPU
    SampledSpectrum li(const Ray &primary_ray, SampledWavelengths &lambda, Sampler *sampler) const;

  private:
    const IntegratorBase *base = nullptr;
    Sampler *samplers = nullptr;
    const Spectrum *cie_y = nullptr;
    int samples_per_pixel = 0;
    int max_depth = 0;
    bool regularize = true;
    bool use_russian_roulette = true;
};
