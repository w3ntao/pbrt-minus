#pragma once

#include <pbrt/base/light.h>
#include <pbrt/euclidean_space/point3.h>
#include <pbrt/gpu/macro.h>

class GPUMemoryAllocator;
class ParameterDictionary;
class Spectrum;
class Transform;

class DistantLight : public LightBase {
  public:
    static DistantLight *create(const Transform &renderFromLight,
                                const ParameterDictionary &parameters,
                                GPUMemoryAllocator &allocator);

    void preprocess(const Bounds3f &scene_bounds) {
        scene_bounds.bounding_sphere(&scene_center, &scene_radius);
    }

    PBRT_CPU_GPU
    pbrt::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                            SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum phi(const SampledWavelengths &lambda) const;

  private:
    const Spectrum *l_emit;
    FloatType scale;
    Point3f scene_center;
    FloatType scene_radius;
};
