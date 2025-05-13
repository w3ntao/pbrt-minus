#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/lights/uniform_infinite_light.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/spectrum_util/rgb_color_space.h>

UniformInfiniteLight *UniformInfiniteLight::create(const Transform &renderFromLight,
                                                   const ParameterDictionary &parameters,
                                                   GPUMemoryAllocator &allocator) {
    auto scale = parameters.get_float("scale", 1.0);

    auto color_space = parameters.global_spectra->rgb_color_space;
    auto cie_y = parameters.global_spectra->cie_y;

    auto L = parameters.get_spectrum("L", SpectrumType::Illuminant, allocator);
    if (L == nullptr) {
        L = color_space->illuminant;
    }

    scale /= L->to_photometric(cie_y);

    auto E_v = parameters.get_float("illuminance", -1);
    if (E_v > 0) {
        // If the scene specifies desired illuminance, first calculate
        // the illuminance from a uniform hemispherical emission
        // of L_v then use this to scale the emission spectrum.
        auto k_e = compute_pi();
        scale *= E_v / k_e;
    }

    auto uniform_infinite_light = allocator.allocate<UniformInfiniteLight>();

    uniform_infinite_light->init(renderFromLight, L, scale);

    return uniform_infinite_light;
}

void UniformInfiniteLight::init(const Transform &renderFromLight, const Spectrum *_Lemit,
                                Real _scale) {
    this->render_from_light = renderFromLight;
    this->light_type = LightType::infinite;

    this->Lemit = _Lemit;
    this->scale = _scale;

    sceneCenter = Point3f(NAN, NAN, NAN);
    sceneRadius = NAN;
}

PBRT_CPU_GPU
SampledSpectrum UniformInfiniteLight::phi(const SampledWavelengths &lambda) const {
    const auto PI = compute_pi();

    return 4 * PI * PI * sqr(sceneRadius) * scale * Lemit->sample(lambda);
}

PBRT_CPU_GPU
pbrt::optional<LightLiSample> UniformInfiniteLight::sample_li(const LightSampleContext &ctx,
                                                              const Point2f &u,
                                                              SampledWavelengths &lambda) const {
    Vector3f wi = sample_uniform_sphere(u);
    auto pdf = uniform_sphere_pdf();

    return LightLiSample(scale * Lemit->sample(lambda), wi, pdf,
                         Interaction(ctx.p() + wi * (2 * sceneRadius)));
}

PBRT_CPU_GPU
Real UniformInfiniteLight::pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                                       bool allow_incomplete_pdf) const {
    if (allow_incomplete_pdf) {
        return 0;
    }

    return uniform_sphere_pdf();
}

PBRT_CPU_GPU
SampledSpectrum UniformInfiniteLight::le(const Ray &ray, const SampledWavelengths &lambda) const {
    return scale * Lemit->sample(lambda);
}

PBRT_CPU_GPU
pbrt::optional<LightLeSample> UniformInfiniteLight::sample_le(const Point2f &u1, const Point2f &u2,
                                                              SampledWavelengths &lambda) const {
    // Sample direction for uniform infinite light ray
    Vector3f w = sample_uniform_sphere(u1);

    // Compute infinite light sample ray
    Frame wFrame = Frame::from_z(-w);

    Point2f cd = sample_uniform_disk_concentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.from_local(Vector3f(cd.x, cd.y, 0));
    Ray ray(pDisk + sceneRadius * -w, w);

    // Compute probabilities for uniform infinite light
    auto pdfPos = 1.0 / (compute_pi() * sqr(sceneRadius));
    auto pdfDir = uniform_sphere_pdf();

    return LightLeSample(scale * Lemit->sample(lambda), ray, pdfPos, pdfDir);
}
