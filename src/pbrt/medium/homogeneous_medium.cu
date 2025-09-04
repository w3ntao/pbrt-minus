#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/medium/homogeneous_medium.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectra/densely_sampled_spectrum.h>
#include <pbrt/spectrum_util/global_spectra.h>

Medium::Medium(const Spectrum *_sigma_a, const Spectrum *_sigma_s, const Real sigma_scale,
               const Real g, GPUMemoryAllocator &allocator) {
    phase = HGPhaseFunction(g);

    sigma_a = allocator.create<DenselySampledSpectrum>(_sigma_a, sigma_scale);
    sigma_s = allocator.create<DenselySampledSpectrum>(_sigma_s, sigma_scale);
}

const Medium *Medium::create(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator) {
    auto build_sigma = [&](const std::string &name, const Real val) -> const Spectrum * {
        if (const auto spectrum =
                parameters.get_spectrum(name, SpectrumType::Unbounded, allocator)) {
            return spectrum;
        }
        return Spectrum::create_constant_spectrum(val, allocator);
    };

    const auto Le = parameters.get_spectrum("Le", SpectrumType::Illuminant, allocator);
    const auto Le_scale = parameters.get_float("Lescale", 1.0);
    if (Le && Le->max_value() > 0 && Le_scale != 0) {
        printf("ERROR: illuminating volume not implemented");
        REPORT_FATAL_ERROR();
    }

    const auto sigma_scale = parameters.get_float("scale", 1.0);
    const auto g = parameters.get_float("g", 0.0);

    auto sigma_a = build_sigma("sigma_a", 1.0);
    auto sigma_s = build_sigma("sigma_s", 1.0);

    return allocator.create<Medium>(sigma_a, sigma_s, sigma_scale, g, allocator);
}

PBRT_CPU_GPU
[[nodiscard]]
SampledSpectrum Medium::sample_sigma_t(const SampledWavelengths &lambda) const {
    return sigma_a->sample(lambda) + sigma_s->sample(lambda);
}
