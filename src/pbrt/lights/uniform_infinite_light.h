#pragma once

#include <pbrt/base/light.h>
#include <pbrt/euclidean_space/bounds3.h>
#include <pbrt/euclidean_space/point3.h>

class GPUMemoryAllocator;
class Spectrum;
class Transform;
class ParameterDictionary;

class UniformInfiniteLight : public LightBase {
  public:
    static UniformInfiniteLight *create(const Transform &renderFromLight,
                                        const ParameterDictionary &parameters,
                                        GPUMemoryAllocator &allocator);

    void preprocess(const Bounds3f &scene_bounds) {
        scene_bounds.bounding_sphere(&sceneCenter, &sceneRadius);
    }

    PBRT_CPU_GPU
    SampledSpectrum phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pbrt::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                            SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Real pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                     bool allow_incomplete_pdf) const;

    PBRT_CPU_GPU
    SampledSpectrum le(const Ray &ray, const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pbrt::optional<LightLeSample> sample_le(const Point2f &u1, const Point2f &u2,
                                            SampledWavelengths &lambda) const;

  private:
    const Spectrum *Lemit;
    Real scale;
    Point3f sceneCenter;
    Real sceneRadius;

    void init(const Transform &renderFromLight, const Spectrum *_Lemit, Real _scale);
};
