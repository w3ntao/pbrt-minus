#pragma once

#include <pbrt/spectrum_util/sampled_spectrum.h>

class Film;
class GPUMemoryAllocator;
class HLBVH;
class ParameterDictionary;
class Ray;
class SampledWavelengths;
class Sampler;

class AmbientOcclusionIntegrator;
class MegakernelPathIntegrator;
class SurfaceNormalIntegrator;

struct IntegratorBase;

class MegakernelIntegrator {
  public:
    enum class Type {
        ambient_occlusion,
        megakernel_path,
        surface_normal,
    };

    explicit MegakernelIntegrator(const AmbientOcclusionIntegrator *ambient_occlusion_integrator)
        : type(Type::ambient_occlusion), ptr(ambient_occlusion_integrator) {}

    explicit MegakernelIntegrator(const MegakernelPathIntegrator *megakernel_path_integrator)
        : type(Type::megakernel_path), ptr(megakernel_path_integrator) {}

    explicit MegakernelIntegrator(const SurfaceNormalIntegrator *surface_normal_integrator)
        : type(Type::surface_normal), ptr(surface_normal_integrator) {}

    static const MegakernelIntegrator *create(const std::string &integrator_name,
                                              const ParameterDictionary &parameters,
                                              const IntegratorBase *integrator_base,
                                              GPUMemoryAllocator &allocator);

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

    PBRT_CPU_GPU
    SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler) const;

    void render(Film *film, const std::string &sampler_type, int samples_per_pixel,
                const IntegratorBase *integrator_base, bool preview) const;

  private:
    Type type;
    const void *ptr = nullptr;
};
