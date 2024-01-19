#pragma once

#include "base/camera.h"

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
    PBRT_GPU PerspectiveCamera(Point3f look_from, Point3f look_at, Vector3f up, double vfov,
                               double aspect, double aperture, double focus_dist) {
        auto PI = acos(-1.0);
        lens_radius = aperture / 2.0f;
        double theta = vfov * PI / 180.0f;
        double half_height = tan(theta / 2.0f);
        double half_width = aspect * half_height;
        origin = look_from;
        w = (look_from - look_at).normalize();

        u = up.cross(w).normalize();
        v = w.cross(u);
        lower_left_corner =
            origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }

    PBRT_GPU Ray get_ray(double s, double t, curandState *local_rand_state) const override {

        Vector3f rd = lens_radius * random_in_unit_disk(local_rand_state);
        Vector3f offset = u * rd.x + v * rd.y;
        return Ray(origin + offset,
                   lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    Point3f origin;
    Point3f lower_left_corner;
    Vector3f horizontal;
    Vector3f vertical;
    Vector3f u, v, w;
    double lens_radius;
};
