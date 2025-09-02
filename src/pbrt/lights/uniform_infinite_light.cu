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
        auto k_e = pbrt::PI;
        scale *= E_v / k_e;
    }

    return allocator.create<UniformInfiniteLight>(renderFromLight, L, scale);
}

PBRT_CPU_GPU
SampledSpectrum UniformInfiniteLight::phi(const SampledWavelengths &lambda) const {
    return 4 * pbrt::PI * pbrt::PI * sqr(sceneRadius) * scale * Lemit->sample(lambda);
}

PBRT_CPU_GPU
pbrt::optional<LightLiSample>
UniformInfiniteLight::sample_li(const LightSampleContext &ctx, const Point2f &u,
                                const SampledWavelengths &lambda) const {
    const auto wi = sample_uniform_sphere(u);
    const auto pdf = uniform_sphere_pdf();

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
pbrt::optional<LightLeSample>
UniformInfiniteLight::sample_le(const Point2f &u1, const Point2f &u2,
                                const SampledWavelengths &lambda) const {
    // Sample direction for uniform infinite light ray
    Vector3f w = sample_uniform_sphere(u1);

    // Compute infinite light sample ray
    Frame wFrame = Frame::from_z(-w);

    Point2f cd = sample_uniform_disk_concentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.from_local(Vector3f(cd.x, cd.y, 0));
    Ray ray(pDisk + sceneRadius * -w, w);

    // Compute probabilities for uniform infinite light
    auto pdfPos = 1.0 / (pbrt::PI * sqr(sceneRadius));
    auto pdfDir = uniform_sphere_pdf();

    return LightLeSample(scale * Lemit->sample(lambda), ray, pdfPos, pdfDir);
}
