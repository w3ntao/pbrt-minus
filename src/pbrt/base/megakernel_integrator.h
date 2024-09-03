#pragma once

#include "pbrt/spectrum_util/sampled_spectrum.h"
#include <optional>
#include <vector>

class Ray;
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

    std::string get_name() const {
        switch (type) {
        case (Type::ambient_occlusion): {
            return "ambientocclusion";
        }

        case (Type::path): {
            return "path";
        }

        case (Type::simple_path): {
            return "simplepath";
        }

        case (Type::surface_normal): {
            return "surfacenormal";
        }
        }

        REPORT_FATAL_ERROR();
        return "";
    }

    void init(const AmbientOcclusionIntegrator *ambient_occlusion_integrator);

    void init(const PathIntegrator *path_integrator);

    void init(const RandomWalkIntegrator *random_walk_integrator);

    void init(const SurfaceNormalIntegrator *surface_normal_integrator);

    void init(const SimplePathIntegrator *simple_path_integrator);

    PBRT_GPU SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler) const;

  private:
    Type type;
    const void *ptr;
};
