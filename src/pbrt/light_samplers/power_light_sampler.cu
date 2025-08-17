#include <pbrt/base/light.h>
#include <pbrt/distribution/distribution_1d.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/light_samplers/power_light_sampler.h>
#include <pbrt/util/hash_map.h>

PowerLightSampler::PowerLightSampler(const Light **_lights, const int _light_num,
                                     GPUMemoryAllocator &allocator)
    : light_num(_light_num), lights(_lights) {
    if (light_num == 0) {
        return;
    }

    std::vector<Real> lights_pmf(light_num);
    const SampledWavelengths lambda = SampledWavelengths::sample_visible(0.5f);
    for (int idx = 0; idx < light_num; ++idx) {
        const auto light = lights[idx];
        auto phi = light->phi(lambda).safe_div(lambda.pdf_as_sampled_spectrum());

        lights_pmf[idx] = phi.average();
    }

    auto light_to_idx = allocator.create<HashMap>(light_num, allocator);
    for (auto idx = 0; idx < light_num; ++idx) {
        light_to_idx->insert(reinterpret_cast<uintptr_t>(lights[idx]), idx);
    }

    light_ptr_to_idx = light_to_idx;
    lights_power_distribution = allocator.create<Distribution1D>(lights_pmf, allocator);
}

PBRT_CPU_GPU
pbrt::optional<SampledLight> PowerLightSampler::sample(const Real u) const {
    if (light_num == 0) {
        return {};
    }

    const auto [light_id, pdf] = lights_power_distribution->sample(u);

    if (DEBUG_MODE) {
        if (light_id >= light_num) {
            REPORT_FATAL_ERROR();
        }
    }

    return SampledLight{.light = lights[light_id], .pdf = pdf};
}

PBRT_CPU_GPU
pbrt::optional<SampledLight> PowerLightSampler::sample(const LightSampleContext &ctx,
                                                       Real u) const {
    return sample(u);
}

PBRT_CPU_GPU
Real PowerLightSampler::pmf(const Light *light) const {
    if (light_num == 0) {
        return 0;
    }

    const auto light_idx = light_ptr_to_idx->lookup(reinterpret_cast<uintptr_t>(light));
    return lights_power_distribution->get_pdf(light_idx);
}
