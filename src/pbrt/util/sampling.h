#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/spectrum_util/constants.h"
#include "pbrt/util/utility_math.h"

PBRT_CPU_GPU
inline FloatType cosine_hemisphere_pdf(FloatType cos_theta) {
    return cos_theta / compute_pi();
}

PBRT_CPU_GPU
inline Point2f sample_uniform_disk_concentric(Point2f u) {
    // Map _u_ to $[-1,1]^2$ and handle degeneracy at the origin
    const auto u_offset = FloatType(2.0) * u - Vector2f(1.0, 1.0);
    if (u_offset == Point2f(0.0, 0.0)) {
        return Point2f(0.0, 0.0);
    }

    // Apply concentric mapping to point
    FloatType r = NAN;
    FloatType theta = NAN;
    FloatType pi_over_4 = compute_pi() / 4.0;

    if (abs(u_offset.x) > abs(u_offset.y)) {
        r = u_offset.x;
        theta = pi_over_4 * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        FloatType pi_over_2 = compute_pi() / 2.0;
        theta = pi_over_2 - pi_over_4 * (u_offset.x / u_offset.y);
    }

    return r * Point2f(std::cos(theta), std::sin(theta));
}

PBRT_CPU_GPU
inline Vector3f sample_uniform_sphere(Point2f u) {
    FloatType z = 1 - 2 * u[0];
    FloatType r = safe_sqrt(1 - sqr(z));

    FloatType phi = 2 * compute_pi() * u[1];

    return {r * std::cos(phi), r * std::sin(phi), z};
}

PBRT_CPU_GPU
inline Vector3f sample_cosine_hemisphere(Point2f u) {
    auto d = sample_uniform_disk_concentric(u);
    auto z = safe_sqrt(1.0 - sqr(d.x) - sqr(d.y));

    return Vector3f(d.x, d.y, z);
}

PBRT_CPU_GPU inline FloatType visible_wavelengths_pdf(FloatType lambda) {
    if (lambda < LAMBDA_MIN || lambda > LAMBDA_MAX) {
        return 0;
    }

    return 0.0039398042f / sqr(std::cosh(0.0072f * (lambda - 538)));
}

PBRT_CPU_GPU inline FloatType sample_visible_wavelengths(FloatType u) {
    return 538 - 138.888889f * std::atanh(0.85691062f - 1.82750197f * u);
}
