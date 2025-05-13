#pragma once

#include <pbrt/base/light.h>
#include <pbrt/gpu/macro.h>

template <typename T>
class Bounds3;

class Distribution2D;
class GPUImage;
class GPUMemoryAllocator;
class Ray;
class RGBColorSpace;
class SampledWavelengths;
class SampledSpectrum;
class ParameterDictionary;

class ImageInfiniteLight : public LightBase {
  public:
    static ImageInfiniteLight *create(const Transform &render_from_light,
                                      const ParameterDictionary &parameters,
                                      GPUMemoryAllocator &allocator);

    void init(const Transform &_render_from_light, const ParameterDictionary &parameters,
              GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    SampledSpectrum le(const Ray &ray, const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pbrt::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                            SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Real pdf_li(const LightSampleContext &ctx, const Vector3f &w,
                     bool allow_incomplete_pdf) const;

    void preprocess(const Bounds3<Real> &scene_bounds);

  private:
    Real scale;
    const RGBColorSpace *color_space;

    const GPUImage *image_ptr;

    const Distribution2D *image_le_distribution;

    Point2i image_resolution;

    Real scene_radius;
    Point3f scene_center;

    PBRT_CPU_GPU SampledSpectrum ImageLe(Point2f uv, const SampledWavelengths &lambda) const;
};
