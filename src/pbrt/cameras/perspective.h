#pragma once

#include <pbrt/base/camera.h>
#include <pbrt/euclidean_space/bounds2.h>
#include <pbrt/scene/parameter_dictionary.h>

class Film;
class FIlter;

class PerspectiveCamera {
  public:
    void init(const Point2i &resolution, const CameraTransform &camera_transform, const Film *_film,
              const Filter *filter, const ParameterDictionary &parameters);

    PBRT_CPU_GPU
    CameraRay generate_ray(const CameraSample &sample, Sampler *sampler) const;

    CameraBase camera_base;

    PBRT_CPU_GPU
    void pdf_we(const Ray &ray, FloatType *pdfPos, FloatType *pdfDir) const;

    PBRT_CPU_GPU
    SampledSpectrum we(const Ray &ray, SampledWavelengths &lambda, Point2f *pRasterOut) const;

    PBRT_CPU_GPU
    pbrt::optional<CameraWiSample> sample_wi(const Interaction &ref, const Point2f u,
                                                  SampledWavelengths &lambda) const;

  private:
    Transform raster_from_screen;
    Transform screen_from_raster;
    Transform screen_from_camera;
    Transform camera_from_raster;

    Vector3f dx_camera;
    Vector3f dy_camera;

    FloatType lens_radius;
    FloatType focal_distance;

    FloatType cosTotalWidth;
    FloatType A;

    const Film *film;
};
