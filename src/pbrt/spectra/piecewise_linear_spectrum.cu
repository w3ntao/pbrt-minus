#include <pbrt/base/spectrum.h>
#include <pbrt/spectra/piecewise_linear_spectrum.h>
#include <pbrt/spectrum_util/spectrum_constants_cie.h>

#include <pbrt/gpu/gpu_memory_allocator.h>

const PiecewiseLinearSpectrum *
PiecewiseLinearSpectrum::create_from_lambdas_values(const std::vector<Real> &cpu_lambdas,
                                                    const std::vector<Real> &cpu_values,
                                                    GPUMemoryAllocator &allocator) {
    auto spectrum = allocator.allocate<PiecewiseLinearSpectrum>();

    spectrum->init_from_lambdas_values(cpu_lambdas, cpu_values, allocator);

    return spectrum;
}

const PiecewiseLinearSpectrum *
PiecewiseLinearSpectrum::create_from_interleaved(const std::vector<Real> &samples,
                                                 bool normalize, const Spectrum *cie_y,
                                                 GPUMemoryAllocator &allocator) {
    auto piecewise_linear_spectrum = allocator.allocate<PiecewiseLinearSpectrum>();

    piecewise_linear_spectrum->init_from_interleaved(samples, normalize, cie_y, allocator);

    return piecewise_linear_spectrum;
}

void PiecewiseLinearSpectrum::init_from_lambdas_values(const std::vector<Real> &cpu_lambdas,
                                                       const std::vector<Real> &cpu_values,
                                                       GPUMemoryAllocator &allocator) {
    if (cpu_lambdas.size() != cpu_values.size()) {
        REPORT_FATAL_ERROR();
    }
    size = cpu_lambdas.size();

    lambdas = allocator.allocate<Real>(size);
    values = allocator.allocate<Real>(size);

    CHECK_CUDA_ERROR(
        cudaMemcpy(lambdas, cpu_lambdas.data(), sizeof(Real) * size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        cudaMemcpy(values, cpu_values.data(), sizeof(Real) * size, cudaMemcpyHostToDevice));
}

void PiecewiseLinearSpectrum::init_from_interleaved(const std::vector<Real> &samples,
                                                    bool normalize, const Spectrum *cie_y,
                                                    GPUMemoryAllocator &allocator) {
    if (samples.size() % 2 != 0) {
        REPORT_FATAL_ERROR();
    }

    uint n = samples.size() / 2;
    std::vector<Real> cpu_lambdas, cpu_values;

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

    this->init_from_lambdas_values(cpu_lambdas, cpu_values, allocator);

    if (normalize) {
        this->scale(CIE_Y_integral / inner_product(cie_y));
    }
}

PBRT_CPU_GPU
SampledSpectrum PiecewiseLinearSpectrum::sample(const SampledWavelengths &lambda) const {
    SampledSpectrum s;
    for (int idx = 0; idx < NSpectrumSamples; ++idx) {
        s[idx] = (*this)(lambda[idx]);
    }

    return s;
}

PBRT_CPU_GPU
Real PiecewiseLinearSpectrum::operator()(Real lambda) const {
    if (size == 0 || lambda < lambdas[0] || lambda > lambdas[size - 1]) {
        return 0;
    }

    // Find offset to largest _lambdas_ below _lambda_ and interpolate
    int o = find_interval(size, [&](int i) { return lambdas[i] <= lambda; });

    auto t = (lambda - lambdas[o]) / (lambdas[o + 1] - lambdas[o]);
    return pbrt::lerp(t, values[o], values[o + 1]);
}

PBRT_CPU_GPU
Real PiecewiseLinearSpectrum::inner_product(const Spectrum *spectrum) const {
    Real sum = 0;
    for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
        sum += (*this)(lambda) * (*spectrum)(lambda);
    }

    return sum;
}
