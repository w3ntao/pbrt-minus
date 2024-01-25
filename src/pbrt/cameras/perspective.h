#pragma once

#include <curand_kernel.h>
#include "pbrt/base/camera.h"
#include "pbrt/euclidean_space/bounds2.h"

PBRT_GPU Vector3f random_in_unit_disk(curandState *local_rand_state) {
    Vector3f p;
    do {
        p = 2.0 * Vector3f(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) -
            Vector3f(1, 1, 0);
    } while (p.dot(p) >= 1.0f);
    return p;
}

class PerspectiveCamera : public Camera {
  public:
    PBRT_GPU PerspectiveCamera(const Point2i &resolution, const CameraTransform &camera_transform)
        : Camera(resolution, camera_transform) {

        // TODO: read fov from parameter
        double fov = 38;
        auto frame_aspect_ratio = double(resolution.x) / double(resolution.y);

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

    PBRT_GPU Ray get_ray(double s, double t, curandState *local_rand_state) const override {
        Vector3f rd = lens_radius * random_in_unit_disk(local_rand_state);
        Vector3f offset = u * rd.x + v * rd.y;
        return Ray(origin + offset,
                   lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    PBRT_GPU Ray generate_ray(const Point2f &sampled_p_film) const override {
        // Compute raster and camera sample positions
        const auto p_film = Point3f(sampled_p_film.x, sampled_p_film.y, 0.0);

        const auto p_camera = camera_from_raster(p_film);

        const auto ray = Ray(Point3f(0, 0, 0), p_camera.to_vector3().normalize());

        return camera_transform.render_from_camera(ray);
    }

  private:
    PBRT_CPU_GPU void print_matrix(const SquareMatrix<4> &matrix) const {
        printf("{\n");
        for (int i = 0; i < 4; ++i) {
            printf("    { ");
            for (int k = 0; k < 4; ++k) {
                printf("%f, ", matrix[i][k]);
            }
            printf("},\n");
        }
        printf("}");
    }

    PBRT_CPU_GPU void print_transform(const Transform &transform) const {
        printf("m:\n");
        print_matrix(transform.m);

        printf("m_inv:\n");
        print_matrix(transform.inv_m);
        printf("\n");
    }

    Transform raster_from_screen;
    Transform screen_from_raster;
    Transform screen_from_camera;
    Transform camera_from_raster;

    Vector3f dx_camera;
    Vector3f dy_camera;

    Point3f origin;
    Point3f lower_left_corner;
    Vector3f horizontal;
    Vector3f vertical;
    Vector3f u, v, w;
    double lens_radius;
};
