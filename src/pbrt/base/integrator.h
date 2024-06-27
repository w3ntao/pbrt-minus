#pragma once

#include <optional>
#include <vector>

#include "pbrt/spectrum_util/sampled_spectrum.h"

class DifferentialRay;
class HLBVH;
class IntegratorBase;
class ParameterDictionary;
class SampledWavelengths;
class Sampler;

class AmbientOcclusionIntegrator;
class PathIntegrator;
class RandomWalkIntegrator;
class SurfaceNormalIntegrator;
class SimplePathIntegrator;

class Integrator {
  public:
    enum class Type {
        ambient_occlusion,
        path,
        random_walk,
        simple_path,
        surface_normal,
    };

    static const Integrator *create(const ParameterDictionary &parameters,
                                    const std::optional<std::string> &_integrator_name,
                                    const IntegratorBase *integrator_base,
                                    std::vector<void *> &gpu_dynamic_pointers);

    void init(const AmbientOcclusionIntegrator *ambient_occlusion_integrator);

    void init(const PathIntegrator *path_integrator);

    void init(const RandomWalkIntegrator *random_walk_integrator);

    void init(const SurfaceNormalIntegrator *surface_normal_integrator);

    void init(const SimplePathIntegrator *simple_path_integrator);

    PBRT_GPU SampledSpectrum li(const DifferentialRay &ray, SampledWavelengths &lambda,
                                Sampler *sampler) const;

  private:
    Type type;
    const void *ptr;
};
