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
SampledSpectrum BSDF::rho(const Vector3f &wo_render, const Real *uc, const Point2f *u,
                           const int sample_count) const {
    const auto wo = render_to_local(wo_render);
    if (wo.z == 0 || sample_count <= 0) {
        return SampledSpectrum(0);
    }

    SampledSpectrum result(0);
    for (int i = 0; i < sample_count; ++i) {
        const auto bs = bxdf.sample_f(wo, uc[i], u[i], TransportMode::Radiance,
                                      BxDFReflTransFlags::All);
        if (bs && bs->pdf > 0) {
            // Keep delta events and layered path weights; do not evaluate
            // f() or pdf() again. Failed samples contribute zero to the mean.
            result += bs->f * bs->wi.abs_cos_theta() / bs->pdf;
        }
    }
    return result / sample_count;
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
DielectricBSDFSample BSDF::sample_dielectric_f(const Vector3f &wo_render, const Point2f &u2,
                                               TransportMode mode,
                                               BxDFReflTransFlags sample_flags) const {
    const auto wo = render_to_local(wo_render);

    if (wo.z == 0 || !(bxdf.flags() & sample_flags)) {
        return {};
    }

    auto samples = bxdf.sample_dielectric_f(wo, u2, mode, sample_flags);
    pbrt::optional<BSDFSample> *branches[] = {&samples.reflection, &samples.transmission};
    Real *direction_pdfs[] = {&samples.reflection_direction_pdf, &samples.transmission_direction_pdf};
    for (int i = 0; i < 2; ++i) {
        auto *branch = branches[i];
        if (!*branch) {
            continue;
        }
        auto &bs = **branch;
        if (!bs.f.is_positive() || !(bs.pdf > 0) || !(*direction_pdfs[i] > 0) || bs.wi.z == 0) {
            branch->reset();
            *direction_pdfs[i] = 0;
            continue;
        }
        bs.wi = local_to_render(bs.wi);
    }
    return samples;
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
