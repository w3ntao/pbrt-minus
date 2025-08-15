#pragma once

#include <cuda/std/tuple>
#include <pbrt/euclidean_space/bounds2.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/euclidean_space/point3.h>
#include <pbrt/euclidean_space/vector3.h>
#include <pbrt/spectrum_util/spectrum_constants_cie.h>
#include <pbrt/util/math.h>

PBRT_CPU_GPU
inline Real cosine_hemisphere_pdf(Real cos_theta) {
    return cos_theta / pbrt::PI;
}

PBRT_CPU_GPU
inline Real sample_linear(Real u, Real a, Real b) {
    if (u == 0 && a == 0) {
        return 0;
    }
    Real x = u * (a + b) / (a + std::sqrt(pbrt::lerp(u, sqr(a), sqr(b))));
    return std::min(x, OneMinusEpsilon);
}

PBRT_CPU_GPU
inline int sample_discrete(const Real *weights, int num_weights, Real u, Real *pmf = nullptr,
                           Real *uRemapped = nullptr) {
    // Handle empty _weights_ for discrete sampling
    if (num_weights == 0) {
        if (pmf != nullptr) {
            *pmf = 0;
        }
        return -1;
    }

    // Compute sum of _weights_
    Real sumWeights = 0;
    for (int idx = 0; idx < num_weights; ++idx) {
        sumWeights += weights[idx];
    }

    // Compute rescaled $u'$ sample
    Real up = u * sumWeights;
    if (up == sumWeights) {
        up = next_float_down(up);
    }

    // Find offset in _weights_ corresponding to $u'$
    int offset = 0;
    Real sum = 0;
    while (sum + weights[offset] <= up) {
        sum += weights[offset++];
    }

    // Compute PMF and remapped _u_ value, if necessary
    if (pmf != nullptr) {
        *pmf = weights[offset] / sumWeights;
    }
    if (uRemapped != nullptr) {
        *uRemapped = std::min((up - sum) / weights[offset], OneMinusEpsilon);
    }

    return offset;
}

PBRT_CPU_GPU
inline Real sample_tent(Real u, Real r) {
    const Real weigits[2] = {0.5, 0.5};

    if (sample_discrete(weigits, 2, u, nullptr, &u) == 0) {
        return -r + r * sample_linear(u, 0, 1);
    }

    return r * sample_linear(u, 1, 0);
}

PBRT_CPU_GPU
inline Real sample_normal(Real u, Real mu = 0, Real sigma = 1) {
    // for GPU code there is a non-zero probability it returns INF (from ErfInv())

    return mu + Sqrt2 * sigma * erf_inv(2 * u - 1);
}

PBRT_CPU_GPU
inline Point2f sample_bilinear(Point2f u, const Real w[4]) {
    Point2f p;
    // Sample $y$ for bilinear marginal distribution
    p.y = sample_linear(u[1], w[0] + w[1], w[2] + w[3]);

    // Sample $x$ for bilinear conditional distribution
    p.x = sample_linear(u[0], pbrt::lerp(p.y, w[0], w[2]), pbrt::lerp(p.y, w[1], w[3]));

    return p;
}

PBRT_CPU_GPU
inline Real bilinear_pdf(Point2f p, const Real w[4]) {
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
    const auto u_offset = Real(2.0) * u - Vector2f(1.0, 1.0);
    if (u_offset == Point2f(0.0, 0.0)) {
        return Point2f(0.0, 0.0);
    }

    // Apply concentric mapping to point
    Real r = NAN;
    Real theta = NAN;
    Real pi_over_4 = pbrt::PI / 4.0;

    if (abs(u_offset.x) > abs(u_offset.y)) {
        r = u_offset.x;
        theta = pi_over_4 * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        Real pi_over_2 = pbrt::PI / 2.0;
        theta = pi_over_2 - pi_over_4 * (u_offset.x / u_offset.y);
    }

    return r * Point2f(std::cos(theta), std::sin(theta));
}

PBRT_CPU_GPU
inline Point2f sample_uniform_disk_polar(const Point2f u) {
    Real r = std::sqrt(u[0]);
    Real theta = 2 * pbrt::PI * u[1];
    return {r * std::cos(theta), r * std::sin(theta)};
}

PBRT_CPU_GPU
inline void sample_uniform_triangle(Real out[3], const Point2f u) {
    Real b0, b1;

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
static void sample_spherical_triangle(Real out[3], Real *pdf, const Point3f v[3], const Point3f p,
                                      const Point2f u) {
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
    Real alpha = angle_between(n_ab, -n_ca);
    Real beta = angle_between(n_bc, -n_ab);
    Real gamma = angle_between(n_ca, -n_bc);

    // Uniformly sample triangle area $A$ to compute $A'$
    Real A_pi = alpha + beta + gamma;
    Real Ap_pi = pbrt::lerp(u[0], pbrt::PI, A_pi);
    if (pdf) {
        Real A = A_pi - pbrt::PI;
        *pdf = (A <= 0) ? 0 : 1 / A;
    }

    // Find $\cos\beta'$ for point along _b_ for sampled area
    Real cosAlpha = std::cos(alpha), sinAlpha = std::sin(alpha);
    Real sinPhi = std::sin(Ap_pi) * cosAlpha - std::cos(Ap_pi) * sinAlpha;
    Real cosPhi = std::cos(Ap_pi) * cosAlpha + std::sin(Ap_pi) * sinAlpha;
    Real k1 = cosPhi + cosAlpha;
    Real k2 = sinPhi - sinAlpha * a.dot(b) /* cos c */;
    Real cosBp = (k2 + (difference_of_products(k2, cosPhi, k1, sinPhi)) * cosAlpha) /
                 ((sum_of_products(k2, sinPhi, k1, cosPhi)) * sinAlpha);

    cosBp = clamp<Real>(cosBp, -1, 1);

    // Sample $c'$ along the arc between $b'$ and $a$
    Real sinBp = safe_sqrt(1 - sqr(cosBp));
    Vector3f cp = cosBp * a + sinBp * gram_schmidt(c, a).normalize();

    // Compute sampled spherical triangle direction and return barycentrics
    Real cosTheta = 1 - u[1] * (1 - cp.dot(b));
    Real sinTheta = safe_sqrt(1 - sqr(cosTheta));
    Vector3f w = cosTheta * b + sinTheta * gram_schmidt(cp, b).normalize();
    // Find barycentric coordinates for sampled direction _w_
    Vector3f e1 = v[1] - v[0], e2 = v[2] - v[0];
    Vector3f s1 = w.cross(e2);
    Real divisor = s1.dot(e1);
    if (divisor == 0) {
        // This happens with triangles that cover (nearly) the whole
        // hemisphere.
        for (int idx = 0; idx < 3; ++idx) {
            out[idx] = 1.0 / 3.0;
        }

        return;
    }

    Real invDivisor = 1 / divisor;
    Vector3f s = p - v[0];
    Real b1 = s.dot(s1) * invDivisor;
    Real b2 = w.dot(s.cross(e1)) * invDivisor;

    // Return clamped barycentrics for sampled direction
    b1 = clamp<Real>(b1, 0, 1);
    b2 = clamp<Real>(b2, 0, 1);
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
    Real z = 1 - 2 * u[0];
    Real r = safe_sqrt(1 - sqr(z));

    Real phi = 2 * pbrt::PI * u[1];

    return {r * std::cos(phi), r * std::sin(phi), z};
}

PBRT_CPU_GPU
inline Real uniform_sphere_pdf() {
    return 1.0 / (4.0 * pbrt::PI);
}

PBRT_CPU_GPU
inline Vector3f sample_cosine_hemisphere(Point2f u) {
    auto d = sample_uniform_disk_concentric(u);
    auto z = safe_sqrt(1.0 - sqr(d.x) - sqr(d.y));

    return Vector3f(d.x, d.y, z);
}

PBRT_CPU_GPU
inline Vector3f SampleUniformCone(Point2f u, Real cosThetaMax) {
    Real cosTheta = (1 - u[0]) + u[0] * cosThetaMax;
    Real sinTheta = safe_sqrt(1 - sqr(cosTheta));

    Real phi = u[1] * 2 * pbrt::PI;
    return SphericalDirection(sinTheta, cosTheta, phi);
}

PBRT_CPU_GPU
inline Real UniformConePDF(Real cosThetaMax) {
    return 1.0 / (2.0 * pbrt::PI * (1.0 - cosThetaMax));
}

PBRT_CPU_GPU
inline Real visible_wavelengths_pdf(Real lambda) {
    if (lambda < LAMBDA_MIN || lambda > LAMBDA_MAX) {
        return 0;
    }

    return 0.0039398042f / sqr(std::cosh(0.0072f * (lambda - 538)));
}

PBRT_CPU_GPU
inline Real sample_visible_wavelengths(Real u) {
    return 538 - 138.888889f * std::atanh(0.85691062f - 1.82750197f * u);
}

PBRT_CPU_GPU
inline Real SampleExponential(Real u, Real a) {
    return -std::log(1 - u) / a;
}

PBRT_CPU_GPU
inline Real power_heuristic(int nf, Real fPdf, int ng, Real gPdf) {
    Real f = nf * fPdf;
    Real g = ng * gPdf;

    if (is_inf(sqr(f))) {
        return 1;
    }

    return sqr(f) / (sqr(f) + sqr(g));
}

PBRT_CPU_GPU
inline Real SmoothStepPDF(Real x, Real a, Real b) {
    if (x < a || x > b) {
        return 0;
    }

    return (2 / (b - a)) * smooth_step(x, a, b);
}

PBRT_CPU_GPU
inline Real SampleSmoothStep(Real u, Real a, Real b) {
    auto cdfMinusU = [=](Real x) -> cuda::std::pair<Real, Real> {
        Real t = (x - a) / (b - a);
        Real P = 2 * pbrt::pow<3>(t) - pbrt::pow<4>(t);
        Real PDeriv = SmoothStepPDF(x, a, b);
        return {P - u, PDeriv};
    };
    return NewtonBisection(a, b, cdfMinusU);
}

PBRT_CPU_GPU
Vector3f sample_henyey_greenstein(const Vector3f &wo, Real g, const Point2f u, Real *pdf);

PBRT_CPU_GPU
// Via Jim Arvo's SphTri.C
Point2f InvertSphericalTriangleSample(const Point3f v[3], const Point3f &p, const Vector3f &w);

PBRT_CPU_GPU
Point2f EqualAreaSphereToSquare(Vector3f v);

PBRT_CPU_GPU
Vector3f EqualAreaSquareToSphere(Point2f p);
