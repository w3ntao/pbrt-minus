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

    const GBuffer &get_gbuffer() const {
        return *gbuffer;
    }

    void render(const std::string &sampler_type, int samples_per_pixel) const;
    void write_images(const std::string &output_filename) const;

    // One thread owns all camera samples and raw attributes for this pixel.
    PBRT_CPU_GPU
    void render_pixel(int pixel_index, Sampler *sampler, int samples_per_pixel) const;

  private:
    PBRT_CPU_GPU
    GBufferPixel evaluate_sample(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler,
                                 int samples_per_pixel) const;

    const IntegratorBase *base;
    const GlobalSpectra *spectra;
    GBuffer *gbuffer;
    Real max_distance;
};
