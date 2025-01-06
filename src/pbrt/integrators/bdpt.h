#pragma once

#include "pbrt/util/macro.h"
#include <vector>

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
    static BDPTIntegrator *create(const ParameterDictionary &parameters,
                                  const IntegratorBase *integrator_base,
                                  const std::string &sampler_type, int samples_per_pixel,
                                  std::vector<void *> &gpu_dynamic_pointers);

    void render(Film *film, uint samples_per_pixel, bool preview);

    PBRT_GPU
    SampledSpectrum li(FilmSample *film_samples, int *film_sample_counter, const Ray &ray,
                       SampledWavelengths &lambda, Sampler *sampler, Vertex *camera_vertices,
                       Vertex *light_vertices) const;

    const IntegratorBase *base;
    Sampler *samplers;

    uint max_depth;

  private:
    bool regularize;
};
