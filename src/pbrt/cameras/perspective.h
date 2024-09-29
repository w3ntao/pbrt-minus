#pragma once

#include "pbrt/base/camera.h"
#include "pbrt/euclidean_space/bounds2.h"
#include "pbrt/scene/parameter_dictionary.h"

class PerspectiveCamera {
  public:
    void init(const Point2i &resolution, const CameraTransform &camera_transform,
              const ParameterDictionary &parameters);

    PBRT_CPU_GPU
    CameraRay generate_ray(const CameraSample &sample, Sampler *sampler) const;

    CameraBase camera_base;

  private:
    Transform raster_from_screen;
    Transform screen_from_raster;
    Transform screen_from_camera;
    Transform camera_from_raster;

    Vector3f dx_camera;
    Vector3f dy_camera;

    FloatType lens_radius;
    FloatType focal_distance;
};
