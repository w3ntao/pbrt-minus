#pragma once

#include "pbrt/base/light.h"
#include "pbrt/util/macro.h"
#include <string>
#include <vector>

template <typename T>
class Bounds3;

class Distribution2D;
class Ray;
class SampledWavelengths;
class SampledSpectrum;
class GPUImage;
class ParameterDictionary;
class RGBColorSpace;

class ImageInfiniteLight : public LightBase {
  public:
    static ImageInfiniteLight *create(const Transform &render_from_light,
                                      const ParameterDictionary &parameters,
                                      std::vector<void *> &gpu_dynamic_pointers);

    void init(const Transform &_render_from_light, const ParameterDictionary &parameters,
              std::vector<void *> &gpu_dynamic_pointers);

    PBRT_GPU
    SampledSpectrum le(const Ray &ray, const SampledWavelengths &lambda) const;

    PBRT_GPU
    cuda::std::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                                 SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum phi(const SampledWavelengths &lambda) const;

    PBRT_GPU
    FloatType pdf_li(const LightSampleContext &ctx, const Vector3f &w,
                     bool allow_incomplete_pdf) const;

    void preprocess(const Bounds3<FloatType> &scene_bounds);

  private:
    FloatType scale;
    const RGBColorSpace *color_space;

    const GPUImage *image_ptr;

    const Distribution2D *image_le_distribution;

    Point2i image_resolution;

    FloatType scene_radius;
    Point3f scene_center;

    PBRT_GPU SampledSpectrum ImageLe(Point2f uv, const SampledWavelengths &lambda) const;
};
