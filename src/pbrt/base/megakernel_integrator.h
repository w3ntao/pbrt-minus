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
class MegakernelPathIntegrator;
class SurfaceNormalIntegrator;

class Integrator {
  public:
    enum class Type {
        ambient_occlusion,
        megakernel_path,
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

        case Type::megakernel_path: {
            return "megakernelpath";
        }

        case Type::surface_normal: {
            return "surfacenormal";
        }
        }

        REPORT_FATAL_ERROR();
        return "";
    }

    PBRT_GPU
    SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler) const;

    void render(Film *film, const std::string &sampler_type, uint samples_per_pixel,
                const IntegratorBase *integrator_base, bool preview) const;

  private:
    void init(const AmbientOcclusionIntegrator *ambient_occlusion_integrator);

    void init(const MegakernelPathIntegrator *megakernel_path_integrator);

    void init(const SurfaceNormalIntegrator *surface_normal_integrator);

    Type type;
    const void *ptr;
};
