#include <pbrt/base/shape.h>
#include <pbrt/base/spectrum.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/lights/diffuse_area_light.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/spectrum_util/rgb_color_space.h>

DiffuseAreaLight::DiffuseAreaLight(const Shape *_shape, const Transform &_render_from_light,
                                   const Medium *medium, const ParameterDictionary &parameters,
                                   GPUMemoryAllocator &allocator)
    : LightBase(LightType::area, _render_from_light, medium), shape(_shape) {
    if (parameters.has_string("filename")) {
        throw std::runtime_error("DiffuseAreaLight::init(): this part is not implemented\n");
    }

    scale = parameters.get_float("scale", 1.0);
    two_sided = parameters.get_bool("twosided", false);

    Lemit = parameters.get_spectrum("L", SpectrumType::Illuminant, allocator);
    if (!Lemit) {
        Lemit = parameters.global_spectra->rgb_color_space->illuminant;
    }

    const auto cie_y = parameters.global_spectra->cie_xyz[1];
    scale /= Lemit->to_photometric(cie_y);

    if (const auto phi_v = parameters.get_float("power", -1.0); phi_v > 0.0) {
        throw std::runtime_error("DiffuseAreaLight::init(): this part is not implemented\n");
    }
}

PBRT_CPU_GPU
SampledSpectrum DiffuseAreaLight::l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                                    const SampledWavelengths &lambda) const {
    // Check for zero emitted radiance from point on area light
    if (!two_sided && n.dot(w) < 0.0) {
        return SampledSpectrum(0.0);
    }

    return scale * Lemit->sample(lambda);
}

PBRT_CPU_GPU
pbrt::optional<LightLiSample> DiffuseAreaLight::sample_li(const LightSampleContext &ctx,
                                                          const Point2f &u,
                                                          const SampledWavelengths &lambda) const {
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

PBRT_CPU_GPU
Real DiffuseAreaLight::pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                              bool allow_incomplete_pdf) const {
    // allow_incomplete_pdf = false
    ShapeSampleContext shapeCtx(ctx.pi, ctx.n, ctx.ns);
    return shape->pdf(shapeCtx, wi);
}

PBRT_CPU_GPU
pbrt::optional<LightLeSample> DiffuseAreaLight::sample_le(Point2f u1, Point2f u2,
                                                          const SampledWavelengths &lambda) const {
    // Sample a point on the area light's _Shape_
    auto ss = shape->sample(u1);
    if (!ss) {
        return {};
    }

    // Sample a cosine-weighted outgoing direction _w_ for area light
    Vector3f w;
    Real pdfDir;

    if (this->two_sided) {
        // Choose side of surface and sample cosine-weighted outgoing direction
        if (u2[0] < 0.5f) {
            u2[0] = std::min(u2[0] * 2, OneMinusEpsilon);
            w = sample_cosine_hemisphere(u2);
        } else {
            u2[0] = std::min((u2[0] - 0.5f) * 2, OneMinusEpsilon);
            w = sample_cosine_hemisphere(u2);
            w.z *= -1;
        }
        pdfDir = cosine_hemisphere_pdf(std::abs(w.z)) / 2;

    } else {
        w = sample_cosine_hemisphere(u2);
        pdfDir = cosine_hemisphere_pdf(w.z);
    }

    if (pdfDir == 0) {
        return {};
    }

    // Return _LightLeSample_ for ray leaving area light
    const Interaction &intr = ss->interaction;

    Frame nFrame = Frame::from_z(intr.n.to_vector3());
    w = nFrame.from_local(w);
    auto const Le = this->l(intr.p(), intr.n, intr.uv, w, lambda);

    auto ray = intr.spawn_ray(w);
    ray.medium = medium;

    return LightLeSample(Le, ray, intr, ss->pdf, pdfDir);
}

PBRT_CPU_GPU
void DiffuseAreaLight::pdf_le(const Interaction &intr, Vector3f w, Real *pdfPos,
                              Real *pdfDir) const {
    *pdfPos = shape->pdf(intr);
    *pdfDir = this->two_sided ? cosine_hemisphere_pdf(intr.n.abs_dot(w)) / 2
                              : cosine_hemisphere_pdf(intr.n.dot(w));
}

PBRT_CPU_GPU
SampledSpectrum DiffuseAreaLight::phi(const SampledWavelengths &lambda) const {
    // TODO: image in DiffuseAreaLight is not implemented

    const auto L = Lemit->sample(lambda) * scale;
    return pbrt::PI * (two_sided ? 2 : 1) * this->shape->area() * L;
}
