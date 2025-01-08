#include "pbrt/base/shape.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/lights/diffuse_area_light.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectra/rgb_illuminant_spectrum.h"
#include "pbrt/spectrum_util/global_spectra.h"

void DiffuseAreaLight::init(const Shape *_shape, const Transform &_render_from_light,
                            const ParameterDictionary &parameters,
                            std::vector<void *> &gpu_dynamic_pointers) {
    if (parameters.has_string("filename")) {
        throw std::runtime_error("DiffuseAreaLight::init(): this part is not implemented\n");
    }

    scale = parameters.get_float("scale", 1.0);
    two_sided = parameters.get_bool("twosided", false);

    l_emit = parameters.get_spectrum("L", SpectrumType::Illuminant, gpu_dynamic_pointers);
    if (l_emit == nullptr) {
        l_emit = parameters.global_spectra->rgb_color_space->illuminant;
    }

    const auto cie_y = parameters.global_spectra->cie_xyz[1];
    scale /= l_emit->to_photometric(cie_y);

    auto phi_v = parameters.get_float("power", -1.0);
    if (phi_v > 0.0) {
        throw std::runtime_error("DiffuseAreaLight::init(): this part is not implemented\n");
    }

    light_type = LightType::area;
    render_from_light = _render_from_light;

    shape = _shape;
}

PBRT_GPU
SampledSpectrum DiffuseAreaLight::l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                                    const SampledWavelengths &lambda) const {
    // Check for zero emitted radiance from point on area light
    if (!two_sided && n.dot(w) < 0.0) {
        return SampledSpectrum(0.0);
    }

    return scale * l_emit->sample(lambda);
}

PBRT_GPU
pbrt::optional<LightLiSample> DiffuseAreaLight::sample_li(const LightSampleContext &ctx,
                                                               const Point2f &u,
                                                               SampledWavelengths &lambda) const {
    // Sample point on shape for _DiffuseAreaLight_
    auto shape_ctx = ShapeSampleContext(ctx.pi, ctx.n, ctx.ns);

    auto ss = shape->sample(shape_ctx, u);

    if (!ss || ss->pdf == 0 || (ss->interaction.p() - ctx.p()).squared_length() == 0) {
        return {};
    }

    Vector3f wi = (ss->interaction.p() - ctx.p()).normalize();
    SampledSpectrum Le = l(ss->interaction.p(), ss->interaction.n, ss->interaction.uv, -wi, lambda);

    if (!Le.is_positive()) {
        return {};
    }

    return LightLiSample(Le, wi, ss->pdf, ss->interaction);
}

PBRT_GPU
FloatType DiffuseAreaLight::pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                                   bool allow_incomplete_pdf) const {
    // allow_incomplete_pdf = false
    ShapeSampleContext shapeCtx(ctx.pi, ctx.n, ctx.ns);
    return shape->pdf(shapeCtx, wi);
}

PBRT_CPU_GPU
SampledSpectrum DiffuseAreaLight::phi(const SampledWavelengths &lambda) const {
    // TODO: image in DiffuseAreaLight is not implemented

    auto L = l_emit->sample(lambda) * scale;
    return compute_pi() * (two_sided ? 2 : 1) * this->shape->area() * L;
}
