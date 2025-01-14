#include <pbrt/samplers/mlt.h>
#include <pbrt/util/sampling.h>

PBRT_CPU_GPU
void MLTSampler::ensure_ready(const int index) {
    // Enlarge _MLTSampler::X_ if necessary and get current $\VEC{X}_i$
    if (index >= LENGTH) {
        REPORT_FATAL_ERROR();
    }
    PrimarySample &sample_i = samples[index];

    // Reset $\VEC{X}_i$ if a large step took place in the meantime
    if (sample_i.last_modification_iteration < last_large_step_iteration) {
        sample_i.value = rng.uniform<FloatType>();
        sample_i.last_modification_iteration = last_large_step_iteration;
    }

    // Apply remaining sequence of mutations to _sample_
    sample_i.backup();
    if (large_step) {
        sample_i.value = rng.uniform<FloatType>();
    } else {
        const int64_t nSmall = current_iteration - sample_i.last_modification_iteration;
        // Apply _nSmall_ small step mutations to $\VEC{X}_i$
        const FloatType effSigma = sigma * std::sqrt((FloatType)nSmall);

        FloatType delta = sample_normal(rng.uniform<FloatType>(), 0, effSigma);
        if (is_inf(delta)) {
            // when random value is 1 or -1, delta evaluated to INF (because of erfinv())
            delta = 0;
            // the value "0" doesn't make much sense,
            // yet I don't think other values would make any more sense here
        }

        sample_i.value += delta;
        sample_i.value -= std::floor(sample_i.value);
        sample_i.value = clamp<FloatType>(sample_i.value, 0, OneMinusEpsilon);
    }

    sample_i.last_modification_iteration = current_iteration;
}

PBRT_CPU_GPU
void MLTSampler::start_iteration() {
    current_iteration++;
    large_step = rng.uniform<FloatType>() < large_step_probability;
}

PBRT_CPU_GPU
void MLTSampler::accept() {
    if (large_step) {
        last_large_step_iteration = current_iteration;
    }
}

PBRT_CPU_GPU
void MLTSampler::reject() {
    for (auto &sample_i : samples) {
        if (sample_i.last_modification_iteration == current_iteration) {
            sample_i.restore();
        }
    }

    --current_iteration;
}

PBRT_CPU_GPU
void MLTSampler::start_stream(int index) {
    stream_index = index;
    sample_index = 0;
}

PBRT_CPU_GPU
FloatType MLTSampler::get_1d() {
    const int index = get_next_index();
    ensure_ready(index);
    return samples[index].value;
}

PBRT_CPU_GPU
Point2f MLTSampler::get_2d() {
    return Point2f(get_1d(), get_1d());
}

PBRT_CPU_GPU
Point2f MLTSampler::get_pixel_2d() {
    return get_2d();
}
