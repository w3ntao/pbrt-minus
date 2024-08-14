#include "pbrt/cameras/perspective.h"
#include "pbrt/base/sampler.h"

void PerspectiveCamera::init(const Point2i &resolution, const CameraTransform &camera_transform,
                             const ParameterDictionary &parameters) {
    camera_base.init(resolution, camera_transform);

    focal_distance = parameters.get_float("focaldistance", 1e6);
    lens_radius = parameters.get_float("lensradius", 0.0);

    auto fov = parameters.get_float("fov", 90.0);

    auto frame_aspect_ratio = FloatType(resolution.x) / FloatType(resolution.y);

    auto screen_window =
        frame_aspect_ratio > 1.0
            ? Bounds2f(Point2f(-frame_aspect_ratio, -1.0), Point2f(frame_aspect_ratio, 1.0))
            : Bounds2f(Point2f(-1.0, -1.0 / frame_aspect_ratio),
                       Point2f(1.0, 1.0 / frame_aspect_ratio));

    auto ndc_from_screen =
        Transform::scale(1.0 / (screen_window.p_max.x - screen_window.p_min.x),
                         1.0 / (screen_window.p_max.y - screen_window.p_min.y), 1.0) *
        Transform::translate(-screen_window.p_min.x, -screen_window.p_max.y, 0.0);

    auto raster_from_ndc = Transform::scale(resolution.x, -resolution.y, 1.0);

    raster_from_screen = raster_from_ndc * ndc_from_screen;

    screen_from_raster = raster_from_screen.inverse();

    screen_from_camera = Transform::perspective(fov, 1e-2, 1000.0);

    camera_from_raster = screen_from_camera.inverse() * screen_from_raster;

    dx_camera =
        camera_from_raster(Point3f(1.0, 0.0, 0.0)) - camera_from_raster(Point3f(0.0, 0.0, 0.0));

    dy_camera =
        camera_from_raster(Point3f(0.0, 1.0, 0.0)) - camera_from_raster(Point3f(0.0, 0.0, 0.0));
}

PBRT_GPU
CameraRay PerspectiveCamera::generate_ray(const CameraSample &sample, Sampler *sampler) const {
    Point3f p_film = Point3f(sample.p_film.x, sample.p_film.y, 0);
    Point3f p_camera = camera_from_raster(p_film);

    Ray ray(Point3f(0, 0, 0), p_camera.to_vector3().normalize());

    if (lens_radius > 0) {
        // Sample point on lens
        Point2f pLens = lens_radius * sample_uniform_disk_concentric(sampler->get_2d());

        // Compute point on plane of focus
        auto ft = focal_distance / ray.d.z;
        Point3f pFocus = ray.at(ft);

        // Update ray for effect of lens
        ray.o = Point3f(pLens.x, pLens.y, 0);
        ray.d = (pFocus - ray.o).normalize();
    }

    return CameraRay(camera_base.camera_transform.render_from_camera(ray));
}
