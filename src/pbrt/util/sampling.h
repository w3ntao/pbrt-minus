#pragma once

#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/util/accurate_arithmetic.h"

PBRT_CPU_GPU
inline double cosine_hemisphere_pdf(double cos_theta) {
    return cos_theta / compute_pi();
}

PBRT_CPU_GPU
inline Point2f sample_uniform_disk_concentric(Point2f u) {
    // Map _u_ to $[-1,1]^2$ and handle degeneracy at the origin
    const auto u_offset = 2.0 * u - Vector2f(1.0, 1.0);
    if (u_offset == Point2f(0.0, 0.0)) {
        return Point2f(0.0, 0.0);
    }
    // Apply concentric mapping to point
    double r = NAN;
    double theta = NAN;
    double pi_over_4 = compute_pi() / 4.0;

    if (abs(u_offset.x) > abs(u_offset.y)) {
        r = u_offset.x;
        theta = pi_over_4 * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        double pi_over_2 = compute_pi() / 2.0;
        theta = pi_over_2 - pi_over_4 * (u_offset.x / u_offset.y);
    }

    return r * Point2f(std::cos(theta), std::sin(theta));
}

PBRT_CPU_GPU
inline Vector3f sample_cosine_hemisphere(Point2f u) {
    auto d = sample_uniform_disk_concentric(u);
    auto z = std::sqrt(1.0 - sqr(d.x) - sqr(d.y));
    return Vector3f(d.x, d.y, z);
}
