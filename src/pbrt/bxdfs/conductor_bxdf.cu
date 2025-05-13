#include <pbrt/bxdfs/conductor_bxdf.h>

PBRT_CPU_GPU
pbrt::optional<BSDFSample> ConductorBxDF::sample_f(Vector3f wo, Real uc, Point2f u,
                                                   TransportMode mode,
                                                   BxDFReflTransFlags sample_flags) const {
    if (!(sample_flags & BxDFReflTransFlags::Reflection)) {
        return {};
    }

    if (mf_distrib.effectively_smooth()) {
        // Sample perfect specular conductor BRDF
        Vector3f wi(-wo.x, -wo.y, wo.z);
        // SampledSpectrum f = FrComplex(AbsCosTheta(wi), eta, k) / AbsCosTheta(wi);

        SampledSpectrum f = FrComplex(wi.abs_cos_theta(), eta, k) / wi.abs_cos_theta();

        return BSDFSample(f, wi, 1, BxDFFlags::SpecularReflection);
    }

    // Sample rough conductor BRDF
    // Sample microfacet normal $\wm$ and reflected direction $\wi$
    if (wo.z == 0) {
        return {};
    }

    Vector3f wm = mf_distrib.sample_wm(wo, u);
    Vector3f wi = Reflect(wo, wm);
    if (!wo.same_hemisphere(wi)) {
        return {};
    }

    // Compute PDF of _wi_ for microfacet reflection
    Real pdf = mf_distrib.pdf(wo, wm) / (4 * wo.abs_dot(wm));

    Real cosTheta_o = wo.abs_cos_theta();
    Real cosTheta_i = wi.abs_cos_theta();

    if (cosTheta_i == 0 || cosTheta_o == 0) {
        return {};
    }

    // Evaluate Fresnel factor _F_ for conductor BRDF
    SampledSpectrum F = FrComplex(wo.abs_dot(wm), eta, k);

    SampledSpectrum f = mf_distrib.D(wm) * F * mf_distrib.G(wo, wi) / (4 * cosTheta_i * cosTheta_o);

    return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);
}

PBRT_CPU_GPU
SampledSpectrum ConductorBxDF::f(Vector3f wo, Vector3f wi, TransportMode mode) const {
    if (!wo.same_hemisphere(wi)) {
        return SampledSpectrum(0);
    }

    if (mf_distrib.effectively_smooth()) {
        return SampledSpectrum(0);
    }

    // Evaluate rough conductor BRDF
    // Compute cosines and $\wm$ for conductor BRDF
    Real cosTheta_o = wo.abs_cos_theta();
    Real cosTheta_i = wi.abs_cos_theta();

    if (cosTheta_i == 0 || cosTheta_o == 0) {
        return SampledSpectrum(0);
    }

    Vector3f wm = wi + wo;
    if (wm.squared_length() == 0) {
        return SampledSpectrum(0);
    }

    wm = wm.normalize();

    // Evaluate Fresnel factor _F_ for conductor BRDF
    SampledSpectrum F = FrComplex(wo.abs_dot(wm), eta, k);

    return mf_distrib.D(wm) * F * mf_distrib.G(wo, wi) / (4 * cosTheta_i * cosTheta_o);
}

PBRT_CPU_GPU
Real ConductorBxDF::pdf(Vector3f wo, Vector3f wi, TransportMode mode,
                             BxDFReflTransFlags sample_flags) const {
    if (!(sample_flags & BxDFReflTransFlags::Reflection)) {
        return 0;
    }

    if (!wo.same_hemisphere(wi)) {
        return 0;
    }

    if (mf_distrib.effectively_smooth()) {
        return 0;
    }

    // Evaluate sampling PDF of rough conductor BRDF
    Vector3f wm = wo + wi;
    if (wm.squared_length() == 0) {
        return 0;
    }

    wm = wm.normalize().face_forward(Vector3f(0, 0, 1));
    return mf_distrib.pdf(wo, wm) / (4 * wo.abs_dot(wm));
}
