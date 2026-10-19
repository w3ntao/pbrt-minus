#pragma once

#include <pbrt/base/ray.h>
#include <pbrt/scene/parameter_dictionary.h>

class Film;
class GreyScaleFilm;
class GPUMemoryAllocator;
class Sampler;
class Spectrum;
struct IntegratorBase;

class MegakernelPathIntegrator {
  public:
    MegakernelPathIntegrator(int samples_per_pixel, const std::string &sampler_type,
                             const ParameterDictionary &parameters, const IntegratorBase *integrator_base,
                             GPUMemoryAllocator &allocator);

    void render(Film *film, GreyScaleFilm &firefly_mask, const std::string &output_filename,
                  bool preview) const;

    PBRT_CPU_GPU
    // Selective PT can match BDPT's scattering-depth limit while legacy callers
    // retain their existing truncation and emissive-surface behavior.
    static SampledSpectrum evaluate_Li_volume(Real *depth, const Ray &primary_ray, SampledWavelengths &lambda,
                                              Sampler *sampler, const IntegratorBase *base, int max_depth,
                                              bool regularize, bool match_bdpt_depth = false);

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
};
