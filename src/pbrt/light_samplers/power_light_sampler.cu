#include <pbrt/base/light.h>
#include <pbrt/distribution/distribution_1d.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/light_samplers/power_light_sampler.h>
#include <pbrt/util/hash_map.h>

const PowerLightSampler *PowerLightSampler::create(const Light **lights, const int light_num,
                                                   GPUMemoryAllocator &allocator) {
    auto power_light_sampler = allocator.allocate<PowerLightSampler>();

    power_light_sampler->lights = nullptr;
    power_light_sampler->lights_power_distribution = nullptr;
    power_light_sampler->light_ptr_to_idx = nullptr;
    power_light_sampler->light_num = light_num;

    if (light_num == 0) {
        return power_light_sampler;
    }

    std::vector<Real> lights_pmf(light_num);
    SampledWavelengths lambda = SampledWavelengths::sample_visible(0.5f);
    for (int idx = 0; idx < light_num; ++idx) {
        auto light = lights[idx];
        auto phi = light->phi(lambda).safe_div(lambda.pdf_as_sampled_spectrum());

        lights_pmf[idx] = phi.average();
    }

    auto light_to_idx = HashMap::create(light_num, allocator);
    for (auto idx = 0; idx < light_num; ++idx) {
        light_to_idx->insert((uintptr_t)lights[idx], idx);
    }

    power_light_sampler->light_num = light_num;
    power_light_sampler->lights = lights;
    power_light_sampler->light_ptr_to_idx = light_to_idx;
    power_light_sampler->lights_power_distribution = Distribution1D::create(lights_pmf, allocator);
    ;

    return power_light_sampler;
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

    auto light_idx = light_ptr_to_idx->lookup((uintptr_t)light);
    return lights_power_distribution->get_pdf(light_idx);
}
