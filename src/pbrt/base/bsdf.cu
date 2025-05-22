#include <pbrt/base/bsdf.h>
#include <pbrt/base/material.h>

PBRT_CPU_GPU
void BSDF::init_bxdf(const Material *material, SampledWavelengths &lambda,
                     const MaterialEvalContext &material_eval_context) {
    bxdf = material->get_bxdf(material_eval_context, lambda);
}

PBRT_CPU_GPU
SampledSpectrum BSDF::f(const Vector3f &woRender, const Vector3f &wiRender,
                        const TransportMode mode) const {
    Vector3f wi = render_to_local(wiRender);
    Vector3f wo = render_to_local(woRender);

    if (wo.z == 0) {
        return SampledSpectrum(0.0);
    }

    return bxdf.f(wo, wi, mode);
}

PBRT_CPU_GPU
pbrt::optional<BSDFSample> BSDF::sample_f(const Vector3f &wo_render, Real u, const Point2f &u2,
                                          TransportMode mode,
                                          BxDFReflTransFlags sample_flags) const {
    const auto wo = render_to_local(wo_render);

    if (wo.z == 0 || !(bxdf.flags() & sample_flags)) {
        return {};
    }

    auto bs = bxdf.sample_f(wo, u, u2, mode, sample_flags);
    if (!bs || !bs->f.is_positive() || bs->pdf == 0 || bs->wi.z == 0) {
        return {};
    }

    bs->wi = local_to_render(bs->wi);
    return bs;
}

PBRT_CPU_GPU
Real BSDF::pdf(const Vector3f &woRender, const Vector3f &wiRender, TransportMode mode,
               BxDFReflTransFlags sampleFlags) const {
    Vector3f wo = render_to_local(woRender);
    Vector3f wi = render_to_local(wiRender);

    if (wo.z == 0) {
        return 0;
    }

    return bxdf.pdf(wo, wi, mode, sampleFlags);
}
