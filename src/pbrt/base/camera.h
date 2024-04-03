#pragma once

#include "pbrt/base/ray.h"
#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/transform.h"
#include "pbrt/spectra/sampled_spectrum.h"

enum class RenderingCoordinateSystem {
    CameraCoordSystem,
    CameraWorldCoordSystem,
    WorldCoordSystem,
};

struct CameraTransform {
    Transform render_from_camera;
    Transform camera_from_render;
    Transform world_from_render;
    Transform render_from_world;

    PBRT_CPU_GPU CameraTransform() {}

    PBRT_CPU_GPU CameraTransform(const Transform &_world_from_camera,
                                 RenderingCoordinateSystem rendering_space) {
        switch (rendering_space) {
        case RenderingCoordinateSystem::CameraCoordSystem: {
            world_from_render = _world_from_camera;
            break;
        }

        case RenderingCoordinateSystem::CameraWorldCoordSystem: {
            // the default option
            auto p_camera = _world_from_camera(Point3f(0, 0, 0));
            world_from_render = Transform::translate(p_camera.x, p_camera.y, p_camera.z);
            break;
        }

        case RenderingCoordinateSystem::WorldCoordSystem: {
            world_from_render = Transform::identity();
            break;
        }
        }

        render_from_world = world_from_render.inverse();
        render_from_camera = render_from_world * _world_from_camera;
        camera_from_render = render_from_camera.inverse();
    }
};

// CameraSample Definition
struct CameraSample {
    Point2f p_film;
    Point2f p_lens;
    double filter_weight = 1;

    PBRT_GPU CameraSample(const Point2f &_p_film, const Point2f &_p_lens, double _filter_weight)
        : p_film(_p_film), p_lens(_p_lens), filter_weight(_filter_weight) {}
};

// CameraRay Definition
struct CameraRay {
    Ray ray;
    SampledSpectrum weight;

    PBRT_GPU CameraRay(const Ray &_ray) : ray(_ray), weight(SampledSpectrum::same_value(1)) {}
};

struct CameraBase {
    Point2i resolution;
    CameraTransform camera_transform;

    PBRT_CPU_GPU CameraBase() {}

    PBRT_CPU_GPU void init(const Point2i _resolution, const CameraTransform _camera_transform) {
        resolution = _resolution;
        camera_transform = _camera_transform;
    }
};

/*
class Camera {
  public:
    Point2i resolution;
    CameraTransform camera_transform;

    PBRT_GPU Camera(const Point2i &_resolution, const CameraTransform &_camera_transform)
        : resolution(_resolution), camera_transform(_camera_transform) {}

    PBRT_GPU virtual ~Camera() {}

    PBRT_GPU virtual CameraRay generate_ray(const CameraSample &sample) const = 0;
};
*/
