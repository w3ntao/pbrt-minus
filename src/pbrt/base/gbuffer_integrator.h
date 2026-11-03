#pragma once

#include <pbrt/film/gbuffer.h>

class GPUMemoryAllocator;
class ParameterDictionary;
class Ray;
class SampledWavelengths;
class Sampler;
struct GlobalSpectra;
struct IntegratorBase;

class GBufferIntegrator {
  public:
    GBufferIntegrator(const ParameterDictionary &parameters, const IntegratorBase *base,
                      GPUMemoryAllocator &allocator);
    GBufferIntegrator(const GlobalSpectra *spectra, const IntegratorBase *base, GPUMemoryAllocator &allocator);

    const GBuffer &get_gbuffer() const {
        return *gbuffer;
    }

    // Computes XYZ Y from an already rendered linear RGB image, then stores
    // the maximum absolute difference to each pixel's 3x3 neighborhood.
    // The input must contain one pixel per G-buffer pixel and be GPU-accessible.
    void set_luminance_guide(const RGB *linear_rgb);
    bool has_luminance_guide() const {
        return luminance_guide_ready;
    }

    void render(const std::string &sampler_type, int samples_per_pixel) const;
    // Replays camera samples with an independent sampler, computing only raw
    // normal, albedo, and coverage. AO, depth, and radiance are not evaluated.
    void render_denoising_guides(const std::string &sampler_type, int samples_per_pixel) const;
    void write_images(const std::string &output_filename) const;

    // One thread owns all camera samples and raw attributes for this pixel.
    PBRT_CPU_GPU
    void render_pixel(int pixel_index, Sampler *sampler, int samples_per_pixel,
                       bool denoising_guides_only = false) const;

  private:
    void render_pass(const std::string &sampler_type, int samples_per_pixel, bool denoising_guides_only) const;

    PBRT_CPU_GPU
    GBufferPixel evaluate_sample(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler,
                                 int samples_per_pixel, bool denoising_guides_only) const;

    const IntegratorBase *base;
    const GlobalSpectra *spectra;
    GBuffer *gbuffer;
    Real max_distance;
    int max_depth = 5;
    bool regularize = false;
    mutable bool luminance_guide_ready = false;
};
