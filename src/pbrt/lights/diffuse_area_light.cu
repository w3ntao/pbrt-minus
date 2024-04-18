#include "pbrt/lights/diffuse_area_light.h"

#include "pbrt/base/shape.h"
#include "pbrt/base/spectrum.h"

#include "pbrt/scene/parameter_dict.h"
#include "pbrt/spectra/rgb_illuminant_spectrum.h"

void DiffuseAreaLight::init(const Transform &render_from_light, const ParameterDict &parameters,
                            const Shape *_shape, const RGBColorSpace &rgb_color_space,
                            const Spectrum &cie_y) {
    // TODO: merge RGBColorSpace and Spectrum cie_y into 1 GlobalVariable
    
    auto rgb_l = parameters.get_rgb("L", std::nullopt);

    RGBIlluminantSpectrum rgb_illuminant_spectrum_l;
    rgb_illuminant_spectrum_l.init(rgb_l, rgb_color_space);

    scale = parameters.get_float("scale", std::optional(1.0));
    two_sided = parameters.get_bool("twosided", std::optional(false));

    if (parameters.has_string("filename")) {
        throw std::runtime_error("DiffuseAreaLight::init(): this part is not implemented\n");
    }

    scale /= rgb_illuminant_spectrum_l.to_photometric(cie_y);

    auto phi_v = parameters.get_float("power", std::optional(-1.0));
    if (phi_v > 0.0) {
        throw std::runtime_error("DiffuseAreaLight::init(): this part is not implemented\n");
    }

    light_base.light_type = LightType::delta_direction;
    light_base.render_from_light = render_from_light;

    shape = _shape;
    area = _shape->area();

    Spectrum spectrum_l;
    spectrum_l.init(&rgb_illuminant_spectrum_l);

    l_emit.init_from_spectrum(spectrum_l);

    printf("wentao: implementing DiffuseAreaLight::init()\n\n");
}

PBRT_GPU
SampledSpectrum DiffuseAreaLight::l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                                    const SampledWavelengths &lambda) const {
    // Check for zero emitted radiance from point on area light
    if (!two_sided && n.dot(w) < 0.0) {
        return SampledSpectrum::same_value(0.f);
    }

    return scale * l_emit.sample(lambda);
}
