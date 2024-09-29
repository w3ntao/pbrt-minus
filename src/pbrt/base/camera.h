#pragma once

#include "pbrt/base/ray.h"
#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/transform.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"
#include <vector>

class Sampler;
class ParameterDictionary;
class PerspectiveCamera;

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

    PBRT_CPU_GPU CameraTransform(const Transform _world_from_camera,
                                 const RenderingCoordinateSystem rendering_space) {
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
    FloatType filter_weight;

    PBRT_CPU_GPU
    CameraSample(const Point2f _p_film, FloatType _filter_weight)
        : p_film(_p_film), filter_weight(_filter_weight) {}
};

// CameraRay Definition
struct CameraRay {
    Ray ray;
    SampledSpectrum weight;

    PBRT_CPU_GPU
    CameraRay() : weight(SampledSpectrum(NAN)) {}

    PBRT_CPU_GPU
    CameraRay(const Ray &_ray) : ray(_ray), weight(SampledSpectrum(1)) {}
};

struct CameraBase {
    Point2i resolution;
    CameraTransform camera_transform;

    Vector3f minPosDifferentialX, minPosDifferentialY;
    Vector3f minDirDifferentialX, minDirDifferentialY;

    PBRT_CPU_GPU CameraBase() {}

    void init(const Point2i _resolution, const CameraTransform &_camera_transform) {
        resolution = _resolution;
        camera_transform = _camera_transform;
    }

    PBRT_CPU_GPU
    void approximate_dp_dxy(const Point3f p, const Normal3f n, const uint samples_per_pixel,
                            Vector3f *dpdx, Vector3f *dpdy) const {
        auto p_camera = camera_transform.camera_from_render(p);
        auto DownZFromCamera =
            Transform::rotate_from_to(p_camera.to_vector3().normalize(), Vector3f(0, 0, 1));

        Point3f pDownZ = DownZFromCamera(p_camera);

        Normal3f nDownZ =
            Normal3f(DownZFromCamera(camera_transform.camera_from_render(n.to_vector3())));
        FloatType d = nDownZ.z * pDownZ.z;

        // Find intersection points for approximated camera differential rays
        Ray xRay(Point3f(0, 0, 0) + minPosDifferentialX, Vector3f(0, 0, 1) + minDirDifferentialX);

        FloatType tx = -(nDownZ.dot(xRay.o.to_vector3()) - d) / nDownZ.dot(xRay.d);
        Ray yRay(Point3f(0, 0, 0) + minPosDifferentialY, Vector3f(0, 0, 1) + minDirDifferentialY);

        FloatType ty = -(nDownZ.dot(yRay.o.to_vector3()) - d) / nDownZ.dot(yRay.d);
        Point3f px = xRay.at(tx);
        Point3f py = yRay.at(ty);

        // Estimate $\dpdx$ and $\dpdy$ in tangent plane at intersection point
        FloatType sppScale =
            std::max<FloatType>(0.125, 1.0 / std::sqrt((FloatType)samples_per_pixel));

        *dpdx = sppScale *
                camera_transform.render_from_camera(DownZFromCamera.apply_inverse(px - pDownZ));
        *dpdy = sppScale *
                camera_transform.render_from_camera(DownZFromCamera.apply_inverse(py - pDownZ));
    }
};

class Camera {
  public:
    enum class Type {
        perspective,
    };

    static Camera *create_perspective_camera(const Point2i &resolution,
                                             const CameraTransform &camera_transform,
                                             const ParameterDictionary &parameters,
                                             std::vector<void *> &gpu_dynamic_pointers);

    void init(const PerspectiveCamera *perspective_camera);

    PBRT_CPU_GPU
    const CameraBase *get_camerabase() const;

    PBRT_CPU_GPU
    CameraRay generate_ray(const CameraSample &sample, Sampler *sampler) const;

    PBRT_CPU_GPU
    void approximate_dp_dxy(const Point3f p, const Normal3f n, uint samples_per_pixel,
                            Vector3f *dpdx, Vector3f *dpdy) const {
        get_camerabase()->approximate_dp_dxy(p, n, samples_per_pixel, dpdx, dpdy);
    }

  private:
    Type type;
    const void *ptr;
};
