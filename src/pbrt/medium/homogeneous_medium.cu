#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/medium/homogeneous_medium.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectra/densely_sampled_spectrum.h>
#include <pbrt/spectrum_util/global_spectra.h>

Medium::Medium(const Spectrum *_sigma_a, const Spectrum *_sigma_s, const Real sigma_scale,
               const Real g, GPUMemoryAllocator &allocator) {
    phase = HGPhaseFunction(g);
    sigma_a = DenselySampledSpectrum::create(_sigma_a, sigma_scale, allocator);
    sigma_s = DenselySampledSpectrum::create(_sigma_s, sigma_scale, allocator);
}

const Medium *Medium::create(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator) {
    auto build_sigma = [&](const std::string &name, const Real val) -> const Spectrum * {
        if (const auto spectrum =
                parameters.get_spectrum(name, SpectrumType::Unbounded, allocator)) {
            return spectrum;
        }
        return Spectrum::create_constant_spectrum(val, allocator);
    };

    auto sigma_a = build_sigma("sigma_a", 1.0);
    auto sigma_s = build_sigma("sigma_s", 1.0);

    auto Le = parameters.get_spectrum("Le", SpectrumType::Illuminant, allocator);
    auto Le_scale = parameters.get_float("Lescale", 1.0);
    if (Le && Le->max_value() > 0 && Le_scale != 0) {
        printf("ERROR: illuminating volume not implemented");
        REPORT_FATAL_ERROR();
    }

    const auto sigma_scale = parameters.get_float("scale", 1.0);
    const auto g = parameters.get_float("g", 0.0);

    auto medium = allocator.allocate<Medium>();
    *medium = Medium(sigma_a, sigma_s, sigma_scale, g, allocator);

    return medium;
}

PBRT_CPU_GPU
[[nodiscard]] SampledSpectrum Medium::sample_sigma_t(const SampledWavelengths &lambda) const {
    const SampledSpectrum sampled_sigma_a = sigma_a->sample(lambda);
    const SampledSpectrum sampled_sigma_s = sigma_s->sample(lambda);
    return sampled_sigma_a + sampled_sigma_s;
}
