#pragma once

#include <pbrt/film/gbuffer.h>

class GPUMemoryAllocator;
struct GlobalSpectra;
struct IntegratorBase;

class GBufferIntegrator {
  public:
    GBufferIntegrator(const GlobalSpectra *spectra, const IntegratorBase *base, GPUMemoryAllocator &allocator);

    const GBuffer &get_gbuffer() const {
        return *gbuffer;
    }

    // Round positive camera SPP down to a perfect square for
    // stratified sampling. Render non-albedo attributes, then albedo, in two
    // passes with the same camera samples. Rho and specular albedo share a
    // local sample count clamped to [4, 16] and rounded down to a square.
    void render(int samples_per_pixel) const;
    void write_images(const std::string &output_filename) const;

  private:
    const IntegratorBase *base;
    const GlobalSpectra *spectra;
    GBuffer *gbuffer;
    Real max_distance;
};
