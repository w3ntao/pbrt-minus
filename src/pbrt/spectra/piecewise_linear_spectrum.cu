#include "pbrt/spectra/piecewise_linear_spectrum.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/spectrum_util/spectrum_constants_cie.h"

PBRT_CPU_GPU
SampledSpectrum PiecewiseLinearSpectrum::sample(const SampledWavelengths &lambda) const {
    SampledSpectrum s;
    for (int idx = 0; idx < NSpectrumSamples; ++idx) {
        s[idx] = (*this)(lambda[idx]);
    }

    return s;
}

void PiecewiseLinearSpectrum::init_from_lambdas_values(const std::vector<FloatType> &cpu_lambdas,
                                                       const std::vector<FloatType> &cpu_values,
                                                       std::vector<void *> &gpu_dynamic_pointers) {
    if (cpu_lambdas.size() != cpu_values.size()) {
        REPORT_FATAL_ERROR();
    }
    size = cpu_lambdas.size();

    cudaMallocManaged(&lambdas, sizeof(FloatType) * size);
    cudaMallocManaged(&values, sizeof(FloatType) * size);
    CHECK_CUDA_ERROR(
        cudaMemcpy(lambdas, cpu_lambdas.data(), sizeof(FloatType) * size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        cudaMemcpy(values, cpu_values.data(), sizeof(FloatType) * size, cudaMemcpyHostToDevice));

    gpu_dynamic_pointers.push_back(lambdas);
    gpu_dynamic_pointers.push_back(values);
}

void PiecewiseLinearSpectrum::init_from_interleaved(const std::vector<FloatType> &samples,
                                                    bool normalize, const Spectrum *cie_y,
                                                    std::vector<void *> &gpu_dynamic_pointers) {
    if (samples.size() % 2 != 0) {
        REPORT_FATAL_ERROR();
    }

    uint n = samples.size() / 2;
    std::vector<FloatType> cpu_lambdas, cpu_values;

    // Extend samples to cover range of visible wavelengths if needed.
    if (samples[0] > LAMBDA_MIN) {
        cpu_lambdas.push_back(LAMBDA_MIN - 1);
        cpu_values.push_back(samples[1]);
    }
    for (size_t i = 0; i < n; ++i) {
        cpu_lambdas.push_back(samples[2 * i]);
        cpu_values.push_back(samples[2 * i + 1]);
    }
    if (cpu_lambdas.back() < LAMBDA_MAX) {
        cpu_lambdas.push_back(LAMBDA_MAX + 1);
        cpu_values.push_back(cpu_values.back());
    }

    if (cpu_lambdas.size() != cpu_values.size()) {
        REPORT_FATAL_ERROR();
    }

    this->init_from_lambdas_values(cpu_lambdas, cpu_values, gpu_dynamic_pointers);

    if (normalize) {
        this->scale(CIE_Y_integral / inner_product(cie_y));
    }
}

PBRT_CPU_GPU
FloatType PiecewiseLinearSpectrum::operator()(FloatType lambda) const {
    if (size == 0 || lambda < lambdas[0] || lambda > lambdas[size - 1]) {
        return 0;
    }

    // Find offset to largest _lambdas_ below _lambda_ and interpolate
    int o = find_interval(size, [&](int i) { return lambdas[i] <= lambda; });

    auto t = (lambda - lambdas[o]) / (lambdas[o + 1] - lambdas[o]);
    return lerp(t, values[o], values[o + 1]);
}

PBRT_CPU_GPU
FloatType PiecewiseLinearSpectrum::inner_product(const Spectrum *spectrum) const {
    FloatType sum = 0;
    for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
        sum += (*this)(lambda) * (*spectrum)(lambda);
    }

    return sum;
}
