#pragma once

#include "pbrt/spectrum_util/sampled_spectrum.h"
#include <vector>

class Film;
class HLBVH;
class IntegratorBase;
class ParameterDictionary;
class Ray;
class SampledWavelengths;
class Sampler;

class AmbientOcclusionIntegrator;
class BDPTIntegrator;
class PathIntegrator;
class RandomWalkIntegrator;
class SurfaceNormalIntegrator;
class SimplePathIntegrator;

class Integrator {
  public:
    enum class Type {
        ambient_occlusion,
        bdpt,
        path,
        random_walk,
        simple_path,
        surface_normal,
    };

    static const Integrator *create(const ParameterDictionary &parameters,
                                    const std::string &integrator_name,
                                    const IntegratorBase *integrator_base,
                                    std::vector<void *> &gpu_dynamic_pointers);

    std::string get_name() const {
        switch (type) {
        case Type::ambient_occlusion: {
            return "ambientocclusion";
        }

        case Type::bdpt: {
            return "bdpt";
        }

        case Type::path: {
            return "path";
        }

        case Type::simple_path: {
            return "simplepath";
        }

        case Type::surface_normal: {
            return "surfacenormal";
        }
        }

        REPORT_FATAL_ERROR();
        return "";
    }

    PBRT_GPU SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler) const;

    void render(Film *film, const std::string &sampler_type, uint samples_per_pixel,
                const IntegratorBase *integrator_base) const;

  private:
    void init(const AmbientOcclusionIntegrator *ambient_occlusion_integrator);

    void init(const BDPTIntegrator *bdpt_integrator);

    void init(const PathIntegrator *path_integrator);

    void init(const RandomWalkIntegrator *random_walk_integrator);

    void init(const SurfaceNormalIntegrator *surface_normal_integrator);

    void init(const SimplePathIntegrator *simple_path_integrator);

    Type type;
    const void *ptr;
};
