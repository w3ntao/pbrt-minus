#include <pbrt/samplers/mlt.h>
#include <pbrt/util/sampling.h>

void MLTSampler::ensure_ready(const int index) {
    // Enlarge _MLTSampler::X_ if necessary and get current $\VEC{X}_i$
    if (static_cast<size_t>(index) >= samples.size()) {
        samples.resize(static_cast<size_t>(index) + 1);
    }
    PrimarySample &sample_i = samples[index];

    // Reset $\VEC{X}_i$ if a large step took place in the meantime
    if (sample_i.last_modification_iteration < last_large_step_iteration) {
        sample_i.value = rng.uniform<Real>();
        sample_i.last_modification_iteration = last_large_step_iteration;
    }

    // Apply remaining sequence of mutations to _sample_
    sample_i.backup();
    if (large_step) {
        sample_i.value = rng.uniform<Real>();
    } else {
        const int64_t nSmall = current_iteration - sample_i.last_modification_iteration;
        // Apply _nSmall_ small step mutations to $\VEC{X}_i$
        const Real effSigma = sigma * std::sqrt((Real)nSmall);

        Real delta = sample_normal(rng.uniform<Real>(), 0, effSigma);
        if (is_inf(delta)) {
            // when random value is 1 or -1, delta evaluated to INF (because of erfinv())
            delta = 0;
            // the value "0" doesn't make much sense,
            // yet I don't think other values would make any more sense here
        }

        sample_i.value += delta;
        sample_i.value -= std::floor(sample_i.value);
        sample_i.value = clamp<Real>(sample_i.value, 0, OneMinusEpsilon);
    }

    sample_i.last_modification_iteration = current_iteration;
}

void MLTSampler::start_iteration() {
    current_iteration++;
    large_step = rng.uniform<Real>() < large_step_probability;
}

void MLTSampler::accept() {
    if (large_step) {
        last_large_step_iteration = current_iteration;
    }
}

void MLTSampler::reject() {
    for (auto &sample_i : samples) {
        if (sample_i.last_modification_iteration == current_iteration) {
            sample_i.restore();
        }
    }

    --current_iteration;
}

void MLTSampler::start_stream(int index) {
    stream_index = index;
    sample_index = 0;
}

Real MLTSampler::get_1d() {
    const int index = get_next_index();
    ensure_ready(index);
    return samples[index].value;
}

Point2f MLTSampler::get_2d() {
    return Point2f(get_1d(), get_1d());
}

Point2f MLTSampler::get_pixel_2d() {
    return get_2d();
}
