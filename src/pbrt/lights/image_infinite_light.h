#pragma once

#include <string>
#include <vector>

#include "pbrt/base/light.h"
#include "pbrt/util/macro.h"

template <typename T>
class Bounds3;

class Ray;
class SampledWavelengths;
class SampledSpectrum;
class GPUImage;
class ParameterDict;
class RGBColorSpace;

class ImageInfiniteLight : LightBase {
  public:
    static ImageInfiniteLight *create(const Transform &_render_from_light,
                                      const std::string &image_filename, FloatType _scale,
                                      const RGBColorSpace *_color_space,
                                      std::vector<void *> &gpu_dynamic_pointers);

    PBRT_GPU
    SampledSpectrum le(const Ray &ray, const SampledWavelengths &lambda) const;

    PBRT_GPU
    cuda::std::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                                 SampledWavelengths &lambda) const;

    void preprocess(const Bounds3<FloatType> &scene_bounds);

  private:
    const GPUImage *image;
    FloatType scale;
    const RGBColorSpace *color_space;

    FloatType scene_radius;
    Point3f scene_center;

    PBRT_GPU SampledSpectrum ImageLe(Point2f uv, const SampledWavelengths &lambda) const;
};
