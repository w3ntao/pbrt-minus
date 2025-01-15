#include <pbrt/base/bsdf.h>
#include <pbrt/base/material.h>

PBRT_CPU_GPU
void BSDF::init_bxdf(const Material *material, SampledWavelengths &lambda,
                     const MaterialEvalContext &material_eval_context) {
    switch (material->get_material_type()) {
    case Material::Type::coated_conductor: {
        bxdf.init(material->get_coated_conductor_bsdf(material_eval_context, lambda));
        return;
    }
    case Material::Type::coated_diffuse: {
        bxdf.init(material->get_coated_diffuse_bsdf(material_eval_context, lambda));
        return;
    }

    case Material::Type::conductor: {
        bxdf.init(material->get_conductor_bsdf(material_eval_context, lambda));
        return;
    }

    case Material::Type::dielectric: {
        bxdf.init(material->get_dielectric_bsdf(material_eval_context, lambda));
        return;
    }

    case Material::Type::diffuse: {
        bxdf.init(material->get_diffuse_bsdf(material_eval_context, lambda));
        return;
    }

    case Material::Type::mix: {
        printf("\nyou should not see MixMaterial here\n\n");
        REPORT_FATAL_ERROR();
    }

    default: {
        printf("\n%s(): there is a Material type not implemented\n");
        REPORT_FATAL_ERROR();
    }
    }

    REPORT_FATAL_ERROR();
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
pbrt::optional<BSDFSample> BSDF::sample_f(const Vector3f &wo_render, FloatType u, const Point2f &u2,
                                          TransportMode mode,
                                          BxDFReflTransFlags sample_flags) const {
    if (bxdf.has_type_null()) {
        REPORT_FATAL_ERROR();
    }

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
FloatType BSDF::pdf(const Vector3f &woRender, const Vector3f &wiRender, TransportMode mode,
                    BxDFReflTransFlags sampleFlags) const {
    Vector3f wo = render_to_local(woRender);
    Vector3f wi = render_to_local(wiRender);

    if (wo.z == 0) {
        return 0;
    }

    return bxdf.pdf(wo, wi, mode, sampleFlags);
}
