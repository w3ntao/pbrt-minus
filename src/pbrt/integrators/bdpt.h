#pragma once

#include <pbrt/gpu/macro.h>

class GPUMemoryAllocator;
class ParameterDictionary;
class Ray;
class SampledSpectrum;
class SampledWavelengths;
class Sampler;

struct IntegratorBase;
struct Vertex;
struct FilmSample;

class BDPTIntegrator {
  public:
    static BDPTIntegrator *create(int samples_per_pixel, const std::string &sampler_type,
                                  const ParameterDictionary &parameters,
                                  const IntegratorBase *integrator_base,
                                  GPUMemoryAllocator &allocator);

    void render(Film *film, uint samples_per_pixel, bool preview);

    PBRT_GPU
    SampledSpectrum li(FilmSample *film_samples, int *film_sample_counter, const Ray &ray,
                       SampledWavelengths &lambda, Sampler *sampler, Vertex *camera_vertices,
                       Vertex *light_vertices) const;

    const IntegratorBase *base;
    Sampler *samplers;

    uint max_depth;
    uint film_sample_size;

  private:
    bool regularize;
};
