#pragma once

#include "pbrt/util/macro.h"
#include <vector>

class BSDF;
class ParameterDictionary;
class Ray;
class SampledSpectrum;
class SampledWavelengths;
class Sampler;
class SurfaceInteraction;

struct IntegratorBase;
struct Vertex;

class BDPTIntegrator {
  public:
    static BDPTIntegrator *create(const ParameterDictionary &parameters,
                                  const IntegratorBase *integrator_base,
                                  const std::string &sampler_type, int samples_per_pixel,
                                  std::vector<void *> &gpu_dynamic_pointers);

    void render(Film *film, uint samples_per_pixel, bool preview);

    PBRT_GPU
    SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler,
                       Vertex *camera_vertices, Vertex *light_vertices) const;

    const IntegratorBase *base;
    Sampler *samplers;

    uint max_depth;
};
