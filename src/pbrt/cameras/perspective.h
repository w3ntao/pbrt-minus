#pragma once

#include <pbrt/cameras/camera_base.h>

class Film;
class Filter;
class ParameterDictionary;

class PerspectiveCamera {
  public:
    PerspectiveCamera(const Point2i &resolution, const CameraTransform &camera_transform,
                      const Film *_film, const Filter *filter,
                      const ParameterDictionary &parameters);

    PBRT_CPU_GPU
    CameraRay generate_ray(const CameraSample &sample, Sampler *sampler) const;

    CameraBase camera_base;

    PBRT_CPU_GPU
    void pdf_we(const Ray &ray, Real *pdfPos, Real *pdfDir) const;

    PBRT_CPU_GPU
    SampledSpectrum we(const Ray &ray, SampledWavelengths &lambda, Point2f *pRasterOut) const;

    PBRT_CPU_GPU
    pbrt::optional<CameraWiSample> sample_wi(const Interaction &ref, const Point2f u,
                                             SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    const CameraBase *get_camerabase() const {
        return &camera_base;
    }

  private:
    Transform raster_from_screen;
    Transform screen_from_raster;
    Transform screen_from_camera;
    Transform camera_from_raster;

    Vector3f dx_camera;
    Vector3f dy_camera;

    Real lens_radius;
    Real focal_distance;

    Real cosTotalWidth;
    Real A;

    const Film *film;
};
