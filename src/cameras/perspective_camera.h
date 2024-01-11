#pragma once

#include "base/camera.h"

__device__ Vector3 random_in_unit_disk(curandState *local_rand_state) {
    Vector3 p;
    do {
        p = 2.0f * Vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) -
            Vector3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class PerspectiveCamera : public Camera {
    public:
        ~PerspectiveCamera() override = default;

        __device__ PerspectiveCamera(Point look_from, Point look_at, Vector3 up, double vfov, double aspect,
                                     double aperture, double focus_dist) {
            auto PI = acos(-1.0);
            lens_radius = aperture / 2.0f;
            double theta = vfov * PI / 180.0f;
            double half_height = tan(theta / 2.0f);
            double half_width = aspect * half_height;
            origin = look_from;
            w = (look_from - look_at).normalize();
            u = cross(up, w).normalize();
            v = cross(w, u);
            lower_left_corner =
                origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
            horizontal = 2.0f * half_width * focus_dist * u;
            vertical = 2.0f * half_height * focus_dist * v;
        }

        __device__ Ray get_ray(double s, double t, curandState *local_rand_state) const override {
            Vector3 rd = lens_radius * random_in_unit_disk(local_rand_state);
            Vector3 offset = u * rd.x + v * rd.y;
            return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
        }

        Point origin;
        Point lower_left_corner;
        Vector3 horizontal;
        Vector3 vertical;
        Vector3 u, v, w;
        double lens_radius;
};
