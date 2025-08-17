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
    ImageInfiniteLight(const Transform &render_from_light, Real _scale,
                       const RGBColorSpace *_color_space, const GPUImage *_image_ptr,
                       GPUMemoryAllocator &allocator);

    static ImageInfiniteLight *create(const Transform &render_from_light,
                                      const ParameterDictionary &parameters,
                                      GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    SampledSpectrum le(const Ray &ray, const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pbrt::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                            SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum phi(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Real pdf_li(const LightSampleContext &ctx, const Vector3f &w, bool allow_incomplete_pdf) const;

    void preprocess(const Bounds3<Real> &scene_bounds);

  private:
    Real scale = NAN;
    const RGBColorSpace *color_space = nullptr;

    const GPUImage *image_ptr = nullptr;
    const Distribution2D *image_le_distribution = nullptr;

    Point2i image_resolution = Point2i(0, 0);

    Real scene_radius = NAN;
    Point3f scene_center = Point3f(NAN, NAN, NAN);

    PBRT_CPU_GPU
    SampledSpectrum ImageLe(Point2f uv, const SampledWavelengths &lambda) const;
};
