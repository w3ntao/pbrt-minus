#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/euclidean_space/point3.h"
#include "pbrt/spectrum_util/spectrum_constants.h"
#include "pbrt/util/utility_math.h"

PBRT_CPU_GPU
inline FloatType cosine_hemisphere_pdf(FloatType cos_theta) {
    return cos_theta / compute_pi();
}

PBRT_CPU_GPU
inline FloatType sample_linear(FloatType u, FloatType a, FloatType b) {
    if (u == 0 && a == 0) {
        return 0;
    }
    FloatType x = u * (a + b) / (a + std::sqrt(lerp(u, sqr(a), sqr(b))));
    return std::min(x, OneMinusEpsilon);
}

PBRT_CPU_GPU
inline Point2f sample_bilinear(Point2f u, const FloatType w[4]) {
    Point2f p;
    // Sample $y$ for bilinear marginal distribution
    p.y = sample_linear(u[1], w[0] + w[1], w[2] + w[3]);

    // Sample $x$ for bilinear conditional distribution
    p.x = sample_linear(u[0], lerp(p.y, w[0], w[2]), lerp(p.y, w[1], w[3]));

    return p;
}

PBRT_CPU_GPU
inline FloatType bilinear_pdf(Point2f p, const FloatType w[4]) {
    if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) {
        return 0;
    }

    if (w[0] + w[1] + w[2] + w[3] == 0) {
        return 1;
    }

    return 4.0 *
           ((1 - p[0]) * (1 - p[1]) * w[0] + p[0] * (1 - p[1]) * w[1] + (1 - p[0]) * p[1] * w[2] +
            p[0] * p[1] * w[3]) /
           (w[0] + w[1] + w[2] + w[3]);
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
inline Point2f sample_uniform_disk_polar(const Point2f u) {
    FloatType r = std::sqrt(u[0]);
    FloatType theta = 2 * compute_pi() * u[1];
    return {r * std::cos(theta), r * std::sin(theta)};
}

PBRT_CPU_GPU
inline void sample_uniform_triangle(FloatType out[3], const Point2f u) {
    FloatType b0, b1;

    if (u[0] < u[1]) {
        b0 = u[0] / 2;
        b1 = u[1] - b0;
    } else {
        b1 = u[1] / 2;
        b0 = u[0] - b1;
    }

    out[0] = b0;
    out[1] = b1;
    out[2] = 1 - b0 - b1;
}

PBRT_CPU_GPU
static void sample_spherical_triangle(FloatType out[3], FloatType *pdf, const Point3f v[3],
                                      const Point3f p, const Point2f u) {
    if (pdf) {
        *pdf = 0.0;
    }
    // Compute vectors _a_, _b_, and _c_ to spherical triangle vertices
    Vector3f a(v[0] - p);
    Vector3f b(v[1] - p);
    Vector3f c(v[2] - p);

    a = a.normalize();
    b = b.normalize();
    c = c.normalize();

    // Compute normalized cross products of all direction pairs
    Vector3f n_ab = a.cross(b);
    Vector3f n_bc = b.cross(c);
    Vector3f n_ca = c.cross(a);

    if (n_ab.squared_length() == 0 || n_bc.squared_length() == 0 || n_ca.squared_length() == 0) {
        return;
    }

    n_ab = n_ab.normalize();
    n_bc = n_bc.normalize();
    n_ca = n_ca.normalize();

    // Find angles $\alpha$, $\beta$, and $\gamma$ at spherical triangle vertices
    FloatType alpha = angle_between(n_ab, -n_ca);
    FloatType beta = angle_between(n_bc, -n_ab);
    FloatType gamma = angle_between(n_ca, -n_bc);

    const auto PI = compute_pi();

    // Uniformly sample triangle area $A$ to compute $A'$
    FloatType A_pi = alpha + beta + gamma;
    FloatType Ap_pi = lerp(u[0], PI, A_pi);
    if (pdf) {
        FloatType A = A_pi - PI;
        *pdf = (A <= 0) ? 0 : 1 / A;
    }

    // Find $\cos\beta'$ for point along _b_ for sampled area
    FloatType cosAlpha = std::cos(alpha), sinAlpha = std::sin(alpha);
    FloatType sinPhi = std::sin(Ap_pi) * cosAlpha - std::cos(Ap_pi) * sinAlpha;
    FloatType cosPhi = std::cos(Ap_pi) * cosAlpha + std::sin(Ap_pi) * sinAlpha;
    FloatType k1 = cosPhi + cosAlpha;
    FloatType k2 = sinPhi - sinAlpha * a.dot(b) /* cos c */;
    FloatType cosBp = (k2 + (difference_of_products(k2, cosPhi, k1, sinPhi)) * cosAlpha) /
                      ((sum_of_products(k2, sinPhi, k1, cosPhi)) * sinAlpha);

    cosBp = clamp<FloatType>(cosBp, -1, 1);

    // Sample $c'$ along the arc between $b'$ and $a$
    FloatType sinBp = safe_sqrt(1 - sqr(cosBp));
    Vector3f cp = cosBp * a + sinBp * gram_schmidt(c, a).normalize();

    // Compute sampled spherical triangle direction and return barycentrics
    FloatType cosTheta = 1 - u[1] * (1 - cp.dot(b));
    FloatType sinTheta = safe_sqrt(1 - sqr(cosTheta));
    Vector3f w = cosTheta * b + sinTheta * gram_schmidt(cp, b).normalize();
    // Find barycentric coordinates for sampled direction _w_
    Vector3f e1 = v[1] - v[0], e2 = v[2] - v[0];
    Vector3f s1 = w.cross(e2);
    FloatType divisor = s1.dot(e1);
    if (divisor == 0) {
        // This happens with triangles that cover (nearly) the whole
        // hemisphere.
        for (uint idx = 0; idx < 3; ++idx) {
            out[idx] = 1.0 / 3.0;
        }

        return;
    }

    FloatType invDivisor = 1 / divisor;
    Vector3f s = p - v[0];
    FloatType b1 = s.dot(s1) * invDivisor;
    FloatType b2 = w.dot(s.cross(e1)) * invDivisor;

    // Return clamped barycentrics for sampled direction
    b1 = clamp<FloatType>(b1, 0, 1);
    b2 = clamp<FloatType>(b2, 0, 1);
    if (b1 + b2 > 1) {
        b1 /= b1 + b2;
        b2 /= b1 + b2;
    }

    out[0] = 1.0 - b1 - b2;
    out[1] = b1;
    out[2] = b2;
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

PBRT_CPU_GPU
inline FloatType visible_wavelengths_pdf(FloatType lambda) {
    if (lambda < LAMBDA_MIN || lambda > LAMBDA_MAX) {
        return 0;
    }

    return 0.0039398042f / sqr(std::cosh(0.0072f * (lambda - 538)));
}

PBRT_CPU_GPU
inline FloatType sample_visible_wavelengths(FloatType u) {
    return 538 - 138.888889f * std::atanh(0.85691062f - 1.82750197f * u);
}
