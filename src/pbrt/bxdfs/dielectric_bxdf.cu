#include "pbrt/bxdfs/dielectric_bxdf.h"
#include "pbrt/euclidean_space/normal3f.h"

PBRT_CPU_GPU
cuda::std::optional<BSDFSample> DielectricBxDF::sample_f(Vector3f wo, FloatType uc, Point2f u,
                                                         TransportMode mode,
                                                         BxDFReflTransFlags sample_flags) const {
    if (eta == 1 || mfDistrib.effectively_smooth()) {
        // Sample perfect specular dielectric BSDF
        // FloatType R = FrDielectric(CosTheta(wo), eta), T = 1 - R;
        FloatType R = FrDielectric(wo.cos_theta(), eta);
        FloatType T = 1 - R;
        // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
        FloatType pr = R;
        FloatType pt = T;

        if (!(sample_flags & BxDFReflTransFlags::Reflection)) {
            pr = 0;
        }

        if (!(sample_flags & BxDFReflTransFlags::Transmission)) {
            pt = 0;
        }

        if (pr == 0 && pt == 0) {
            return {};
        }

        if (uc < pr / (pr + pt)) {
            // Sample perfect specular dielectric BRDF
            Vector3f wi(-wo.x, -wo.y, wo.z);
            // SampledSpectrum fr(R / AbsCosTheta(wi));
            auto fr = SampledSpectrum(R / wi.abs_cos_theta());
            return BSDFSample(fr, wi, pr / (pr + pt), BxDFFlags::SpecularReflection);

        } else {
            // Sample perfect specular dielectric BTDF
            // Compute ray direction for specular transmission
            Vector3f wi;
            FloatType etap;
            bool valid = refract(wo, Normal3f(0, 0, 1), eta, &etap, &wi);
            if (!valid) {
                return {};
            }

            auto ft = SampledSpectrum(T / wi.abs_cos_theta());
            // Account for non-symmetry with transmission to different medium
            if (mode == TransportMode::Radiance) {
                ft /= sqr(etap);
            }

            return BSDFSample(ft, wi, pt / (pr + pt), BxDFFlags::SpecularTransmission, etap);
        }

    } else {
        // Sample rough dielectric BSDF
        Vector3f wm = mfDistrib.sample_wm(wo, u);
        FloatType R = FrDielectric(wo.dot(wm), eta);
        FloatType T = 1 - R;
        // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
        FloatType pr = R, pt = T;
        if (!(sample_flags & BxDFReflTransFlags::Reflection)) {
            pr = 0;
        }

        if (!(sample_flags & BxDFReflTransFlags::Transmission)) {
            pt = 0;
        }

        if (pr == 0 && pt == 0) {
            return {};
        }

        FloatType _pdf;
        if (uc < pr / (pr + pt)) {
            // Sample reflection at rough dielectric interface
            Vector3f wi = Reflect(wo, wm);
            if (!wo.same_hemisphere(wi)) {
                return {};
            }
            // Compute PDF of rough dielectric reflection

            _pdf = mfDistrib.pdf(wo, wm) / (4 * wo.abs_dot(wm)) * pr / (pr + pt);
            SampledSpectrum f = SampledSpectrum(mfDistrib.D(wm) * mfDistrib.G(wo, wi) * R /
                                                (4 * wi.cos_theta() * wo.cos_theta()));

            return BSDFSample(f, wi, _pdf, BxDFFlags::GlossyReflection);

        } else {
            // Sample transmission at rough dielectric interface
            FloatType etap;
            Vector3f wi;
            bool tir = !refract(wo, (Normal3f)wm, eta, &etap, &wi);

            if (wo.same_hemisphere(wi) || wi.z == 0 || tir) {
                return {};
            }

            // Compute PDF of rough dielectric transmission
            FloatType denom = sqr(wi.dot(wm) + wo.dot(wm) / etap);

            FloatType dwm_dwi = wi.abs_dot(wm) / denom;
            _pdf = mfDistrib.pdf(wo, wm) * dwm_dwi * pt / (pr + pt);

            // Evaluate BRDF and return _BSDFSample_ for rough transmission
            auto ft = SampledSpectrum(
                T * mfDistrib.D(wm) * mfDistrib.G(wo, wi) *
                std::abs(wi.dot(wm) * wo.dot(wm) / (wi.cos_theta() * wo.cos_theta() * denom)));

            // Account for non-symmetry with transmission to different medium
            if (mode == TransportMode::Radiance) {
                ft /= sqr(etap);
            }

            return BSDFSample(ft, wi, _pdf, BxDFFlags::GlossyTransmission, etap);
        }
    }
}

PBRT_CPU_GPU
SampledSpectrum DielectricBxDF::f(Vector3f wo, Vector3f wi, TransportMode mode) const {
    if (eta == 1 || mfDistrib.effectively_smooth()) {
        return SampledSpectrum(0.0);
    }

    // Evaluate rough dielectric BSDF
    // Compute generalized half vector _wm_
    FloatType cosTheta_o = wo.cos_theta();
    FloatType cosTheta_i = wi.cos_theta();

    bool reflect = cosTheta_i * cosTheta_o > 0;
    float etap = 1;
    if (!reflect) {
        etap = cosTheta_o > 0 ? eta : (1 / eta);
    }

    Vector3f wm = wi * etap + wo;

    if (cosTheta_i == 0 || cosTheta_o == 0 || wm.squared_length() == 0) {
        return SampledSpectrum(0.0);
    }

    wm = wm.face_forward(Vector3f(0, 0, 1)).normalize();

    // Discard backfacing microfacets
    if (wm.dot(wi) * cosTheta_i < 0 || wm.dot(wo) * cosTheta_o < 0) {
        return SampledSpectrum(0.0);
    }

    FloatType F = FrDielectric(wo.dot(wm), eta);

    if (reflect) {
        // Compute reflection at rough dielectric interface
        return SampledSpectrum(mfDistrib.D(wm) * mfDistrib.G(wo, wi) * F /
                               std::abs(4 * cosTheta_i * cosTheta_o));
    }

    // Compute transmission at rough dielectric interface
    FloatType denom = sqr(wi.dot(wm) + wo.dot(wm) / etap) * cosTheta_i * cosTheta_o;

    FloatType ft =
        mfDistrib.D(wm) * (1 - F) * mfDistrib.G(wo, wi) * std::abs(wi.dot(wm) * wo.dot(wm) / denom);

    // Account for non-symmetry with transmission to different medium
    if (mode == TransportMode::Radiance) {
        ft /= sqr(etap);
    }

    return SampledSpectrum(ft);
}

PBRT_CPU_GPU
FloatType DielectricBxDF::pdf(Vector3f wo, Vector3f wi, TransportMode mode,
                              BxDFReflTransFlags sampleFlags) const {
    if (eta == 1 || mfDistrib.effectively_smooth()) {
        return 0;
    }

    // Evaluate sampling PDF of rough dielectric BSDF
    // Compute generalized half vector _wm_
    FloatType cosTheta_o = wo.cos_theta();
    FloatType cosTheta_i = wi.cos_theta();

    bool reflect = cosTheta_i * cosTheta_o > 0;
    float etap = 1;
    if (!reflect) {
        etap = cosTheta_o > 0 ? eta : (1 / eta);
    }

    Vector3f wm = wi * etap + wo;

    if (cosTheta_i == 0 || cosTheta_o == 0 || wm.squared_length() == 0) {
        return {};
    }

    wm = wm.face_forward(Vector3f(0, 0, 1)).normalize();

    // Discard backfacing microfacets
    if (wm.dot(wi) * cosTheta_i < 0 || wm.dot(wo) * cosTheta_o < 0) {
        return {};
    }

    // Determine Fresnel reflectance of rough dielectric boundary
    FloatType R = FrDielectric(wo.dot(wm), eta);
    FloatType T = 1 - R;

    // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
    FloatType pr = R, pt = T;
    if (!(sampleFlags & BxDFReflTransFlags::Reflection)) {
        pr = 0;
    }

    if (!(sampleFlags & BxDFReflTransFlags::Transmission)) {
        pt = 0;
    }

    if (pr == 0 && pt == 0) {
        return {};
    }

    // Return PDF for rough dielectric
    FloatType pdf;
    if (reflect) {
        // Compute PDF of rough dielectric reflection
        pdf = mfDistrib.pdf(wo, wm) / (4 * wo.abs_dot(wm)) * pr / (pr + pt);

    } else {
        // Compute PDF of rough dielectric transmission
        FloatType denom = sqr(wi.dot(wm) + wo.dot(wm) / etap);
        FloatType dwm_dwi = wi.abs_dot(wm) / denom;
        pdf = mfDistrib.pdf(wo, wm) * dwm_dwi * pt / (pr + pt);
    }
    return pdf;
}
