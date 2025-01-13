#include <pbrt/samplers/mlt.h>
#include <pbrt/util/sampling.h>

PBRT_CPU_GPU
void MLTSampler::EnsureReady(const int index) {
    // Enlarge _MLTSampler::X_ if necessary and get current $\VEC{X}_i$
    if (index >= LENGTH) {
        REPORT_FATAL_ERROR();
    }
    PrimarySample &X_i = X[index];

    // Reset $\VEC{X}_i$ if a large step took place in the meantime
    if (X_i.lastModificationIteration < lastLargeStepIteration) {
        X_i.value = rng.uniform<FloatType>();
        X_i.lastModificationIteration = lastLargeStepIteration;
    }

    // Apply remaining sequence of mutations to _sample_
    X_i.Backup();
    if (largeStep) {
        X_i.value = rng.uniform<FloatType>();
    } else {
        const int64_t nSmall = currentIteration - X_i.lastModificationIteration;
        // Apply _nSmall_ small step mutations to $\VEC{X}_i$
        const FloatType effSigma = sigma * std::sqrt((FloatType)nSmall);

        FloatType delta = sample_normal(rng.uniform<FloatType>(), 0, effSigma);
        if (is_inf(delta)) {
            // when random value is 1 or -1, delta evaluated to INF (because of erfinv())
            delta = 0;
            // the value "0" doesn't make much sense,
            // yet I don't think other values would make any more sense here
        }

        X_i.value += delta;
        X_i.value -= std::floor(X_i.value);
        X_i.value = clamp<FloatType>(X_i.value, 0, OneMinusEpsilon);
    }

    X_i.lastModificationIteration = currentIteration;
}

PBRT_CPU_GPU
void MLTSampler::StartIteration() {
    currentIteration++;
    largeStep = rng.uniform<FloatType>() < largeStepProbability;
}

PBRT_CPU_GPU
void MLTSampler::Accept() {
    if (largeStep) {
        lastLargeStepIteration = currentIteration;
    }
}

PBRT_CPU_GPU
void MLTSampler::Reject() {
    for (auto &X_i : X) {
        if (X_i.lastModificationIteration == currentIteration) {
            X_i.Restore();
        }
    }

    --currentIteration;
}

PBRT_CPU_GPU
void MLTSampler::StartStream(int index) {
    streamIndex = index;
    sampleIndex = 0;
}

PBRT_CPU_GPU
FloatType MLTSampler::Get1D() {
    int index = GetNextIndex();
    EnsureReady(index);
    return X[index].value;
}

PBRT_CPU_GPU
Point2f MLTSampler::Get2D() {
    return Point2f(Get1D(), Get1D());
}

PBRT_CPU_GPU
Point2f MLTSampler::GetPixel2D() {
    return Get2D();
}
