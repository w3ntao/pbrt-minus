#pragma once

#include "pbrt/base/light.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/euclidean_space/point3.h"
#include <vector>

class Spectrum;
class Transform;
class ParameterDictionary;

class UniformInfiniteLight : public LightBase {
  public:
    static UniformInfiniteLight *create(const Transform &renderFromLight,
                                        const ParameterDictionary &parameters,
                                        std::vector<void *> &gpu_dynamic_pointers);

    void preprocess(const Bounds3f &scene_bounds) {
        scene_bounds.bounding_sphere(&sceneCenter, &sceneRadius);
    }

    PBRT_CPU_GPU
    SampledSpectrum phi(const SampledWavelengths &lambda) const;

    PBRT_GPU
    pbrt::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                                 SampledWavelengths &lambda) const;

    PBRT_GPU
    FloatType pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                     bool allow_incomplete_pdf) const;

    PBRT_GPU
    SampledSpectrum le(const Ray &ray, const SampledWavelengths &lambda) const;

  private:
    const Spectrum *Lemit;
    FloatType scale;
    Point3f sceneCenter;
    FloatType sceneRadius;

    void init(const Transform &renderFromLight, const Spectrum *_Lemit, FloatType _scale);
};
