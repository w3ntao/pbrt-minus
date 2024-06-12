#include "pbrt/lights/diffuse_area_light.h"

#include "pbrt/base/shape.h"
#include "pbrt/base/spectrum.h"

#include "pbrt/gpu/global_variable.h"

#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectra/rgb_illuminant_spectrum.h"

void DiffuseAreaLight::init(const Transform &_render_from_light,
                            const ParameterDictionary &parameters, const Shape *_shape,
                            const GPU::GlobalVariable *global_variable) {
    auto rgb_l = parameters.get_rgb("L", std::nullopt);

    auto rgb_illuminant_spectrum_l = RGBIlluminantSpectrum(rgb_l, global_variable->rgb_color_space);

    scale = parameters.get_float("scale", 1.0);
    two_sided = parameters.get_bool("twosided", false);

    if (parameters.has_string("filename")) {
        throw std::runtime_error("DiffuseAreaLight::init(): this part is not implemented\n");
    }

    const auto cie_y = global_variable->cie_xyz[1];
    scale /= rgb_illuminant_spectrum_l.to_photometric(cie_y);

    auto phi_v = parameters.get_float("power", -1.0);
    if (phi_v > 0.0) {
        throw std::runtime_error("DiffuseAreaLight::init(): this part is not implemented\n");
    }

    light_type = LightType::delta_direction;
    render_from_light = _render_from_light;

    shape = _shape;

    Spectrum spectrum_l;
    spectrum_l.init(&rgb_illuminant_spectrum_l);

    l_emit.init_from_spectrum(&spectrum_l);
}

PBRT_GPU
SampledSpectrum DiffuseAreaLight::l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                                    const SampledWavelengths &lambda) const {
    // Check for zero emitted radiance from point on area light
    if (!two_sided && n.dot(w) < 0.0) {
        return SampledSpectrum(0.0);
    }

    return scale * l_emit.sample(lambda);
}

PBRT_GPU
cuda::std::optional<LightLiSample> DiffuseAreaLight::sample_li(const LightSampleContext &ctx,
                                                               const Point2f &u,
                                                               SampledWavelengths &lambda) const {
    // Sample point on shape for _DiffuseAreaLight_
    auto shape_ctx = ShapeSampleContext{
        .pi = ctx.pi,
        .n = ctx.n,
        .ns = ctx.ns,
    };

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
