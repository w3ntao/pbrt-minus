#pragma once

#include <pbrt/euclidean_space/normal3f.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/euclidean_space/vector3.h>
#include <pbrt/gpu/macro.h>
#include <pbrt/spectrum_util/sampled_spectrum.h>
#include <pbrt/util/complex.h>
#include <pbrt/util/sampling.h>

PBRT_CPU_GPU
inline Vector3f Reflect(const Vector3f wo, const Vector3f n) {
    return -wo + 2 * wo.dot(n) * n;
}

PBRT_CPU_GPU
inline bool refract(Vector3f wi, Normal3f n, Real eta, Real *etap, Vector3f *wt) {
    Real cosTheta_i = n.dot(wi);

    // Potentially flip interface orientation for Snell's law
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
        n = -n;
    }

    // Compute $\cos\,\theta_\roman{t}$ using Snell's law
    Real sin2Theta_i = std::max<Real>(0, 1 - sqr(cosTheta_i));
    Real sin2Theta_t = sin2Theta_i / sqr(eta);
    // Handle total internal reflection case
    if (sin2Theta_t >= 1) {
        return false;
    }

    Real cosTheta_t = std::sqrt(1 - sin2Theta_t);

    *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * n.to_vector3();
    // Provide relative IOR along ray to caller
    if (etap) {
        *etap = eta;
    }

    return true;
}

PBRT_CPU_GPU
inline Real HenyeyGreenstein(Real cosTheta, Real g) {
    // The Henyey-Greenstein phase function isn't suitable for |g| \approx
    // 1 so we clamp it before it becomes numerically instable. (It's an
    // analogous situation to BSDFs: if the BSDF is perfectly specular, one
    // should use one based on a Dirac delta distribution rather than a
    // very smooth microfacet distribution...)
    g = clamp<Real>(g, -.99, .99);
    Real denom = 1 + sqr(g) + 2 * g * cosTheta;
    return 1.0 / (4.0 * pbrt::PI) * (1 - sqr(g)) / (denom * safe_sqrt(denom));
}

PBRT_CPU_GPU inline Real FrComplex(Real cosTheta_i, pbrt::complex<Real> eta) {
    using Complex = pbrt::complex<Real>;
    cosTheta_i = clamp<Real>(cosTheta_i, 0, 1);

    // Compute complex $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    Real sin2Theta_i = 1 - sqr(cosTheta_i);
    Complex sin2Theta_t = sin2Theta_i / (eta * eta);

    // Complex cosTheta_t = pbrt::sqrt(1 - sin2Theta_t);
    Complex cosTheta_t = (1 - sin2Theta_t).sqrt();

    Complex r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    Complex r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);

    return (r_parl.norm() + r_perp.norm()) / 2;
}

PBRT_CPU_GPU inline SampledSpectrum FrComplex(Real cosTheta_i, SampledSpectrum eta,
                                              SampledSpectrum k) {
    SampledSpectrum result;
    for (int i = 0; i < NSpectrumSamples; ++i) {
        result[i] = FrComplex(cosTheta_i, pbrt::complex<Real>(eta[i], k[i]));
    }

    return result;
}

PBRT_CPU_GPU
inline Real FrDielectric(Real cosTheta_i, Real eta) {
    cosTheta_i = clamp<Real>(cosTheta_i, -1, 1);
    // Potentially flip interface orientation for Fresnel equations
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
    }

    // Compute $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    Real sin2Theta_i = 1 - sqr(cosTheta_i);
    Real sin2Theta_t = sin2Theta_i / sqr(eta);
    if (sin2Theta_t >= 1) {
        return 1.0;
    }

    Real cosTheta_t = safe_sqrt(1 - sin2Theta_t);

    Real r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    Real r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    return (sqr(r_parl) + sqr(r_perp)) / 2;
}

// TrowbridgeReitzDistribution Definition
class TrowbridgeReitzDistribution {
  public:
    // TrowbridgeReitzDistribution Public Methods
    PBRT_CPU_GPU
    TrowbridgeReitzDistribution() : alpha_x(NAN), alpha_y(NAN) {};

    PBRT_CPU_GPU
    TrowbridgeReitzDistribution(Real ax, Real ay) : alpha_x(ax), alpha_y(ay) {
        if (!effectively_smooth()) {
            // If one direction has some roughness, then the other can't
            // have zero (or very low) roughness; the computation of |e| in
            // D() blows up in that case.
            alpha_x = std::max<Real>(alpha_x, 1e-4f);
            alpha_y = std::max<Real>(alpha_y, 1e-4f);
        }
    }

    PBRT_CPU_GPU
    inline Real D(const Vector3f wm) const {
        // FloatType tan2Theta = Tan2Theta(wm);

        Real tan2Theta = wm.tan2_theta();

        if (is_inf(tan2Theta)) {
            return 0;
        }

        Real cos4Theta = sqr(wm.cos2_theta());

        if (cos4Theta < 1e-16f) {
            return 0;
        }

        Real e = tan2Theta * (sqr(wm.cos_phi() / alpha_x) + sqr(wm.sin_phi() / alpha_y));
        return 1 / (pbrt::PI * alpha_x * alpha_y * cos4Theta * sqr(1 + e));
    }

    PBRT_CPU_GPU
    bool effectively_smooth() const {
        return std::max(alpha_x, alpha_y) < 1e-3f;
    }

    PBRT_CPU_GPU
    Real G1(Vector3f w) const {
        return 1 / (1 + Lambda(w));
    }

    PBRT_CPU_GPU
    Real Lambda(Vector3f w) const {
        Real tan2Theta = w.tan2_theta();
        if (is_inf(tan2Theta)) {
            return 0;
        }

        Real alpha2 = sqr(w.cos_phi() * alpha_x) + sqr(w.sin_phi() * alpha_y);
        return (std::sqrt(1 + alpha2 * tan2Theta) - 1) / 2;
    }

    PBRT_CPU_GPU
    Real G(Vector3f wo, Vector3f wi) const {
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }

    PBRT_CPU_GPU
    Real D(Vector3f w, Vector3f wm) const {
        return G1(w) / w.abs_cos_theta() * D(wm) * w.abs_dot(wm);
    }

    PBRT_CPU_GPU
    Real pdf(Vector3f w, Vector3f wm) const {
        return D(w, wm);
    }

    PBRT_CPU_GPU
    Vector3f sample_wm(Vector3f w, Point2f u) const {
        // Transform _w_ to hemispherical configuration
        Vector3f wh = Vector3f(alpha_x * w.x, alpha_y * w.y, w.z).normalize();
        if (wh.z < 0) {
            wh = -wh;
        }

        // Find orthonormal basis for visible normal sampling
        Vector3f T1 =
            (wh.z < 0.99999f) ? Vector3f(0, 0, 1).cross(wh).normalize() : Vector3f(1, 0, 0);

        Vector3f T2 = wh.cross(T1);

        // Generate uniformly distributed points on the unit disk
        Point2f p = sample_uniform_disk_polar(u);

        // Warp hemispherical projection for visible normal sampling
        Real h = std::sqrt(1 - sqr(p.x));
        p.y = pbrt::lerp((1 + wh.z) / 2, h, p.y);

        // Reproject to hemisphere and transform normal to ellipsoid configuration
        Real pz = std::sqrt(std::max<Real>(0, 1 - p.to_vector2f().length_squared()));

        Vector3f nh = p.x * T1 + p.y * T2 + pz * wh;
        return Vector3f(alpha_x * nh.x, alpha_y * nh.y, std::max<Real>(1e-6f, nh.z)).normalize();
    }

    PBRT_CPU_GPU
    static Real RoughnessToAlpha(Real roughness) {
        return std::sqrt(roughness);
    }

    PBRT_CPU_GPU
    void regularize() {
        if (alpha_x < 0.3) {
            alpha_x = clamp(2 * alpha_x, Real(0.1), Real(0.3));
        }

        if (alpha_y < 0.3) {
            alpha_y = clamp(2 * alpha_y, Real(0.1), Real(0.3));
        }
    }

  private:
    Real alpha_x;
    Real alpha_y;
};
