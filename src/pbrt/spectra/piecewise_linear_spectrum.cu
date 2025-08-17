#include <pbrt/base/spectrum.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectra/piecewise_linear_spectrum.h>
#include <pbrt/spectrum_util/spectrum_constants_cie.h>

PiecewiseLinearSpectrum::PiecewiseLinearSpectrum(const std::vector<Real> &samples, bool normalize,
                                                 const Spectrum *cie_y,
                                                 GPUMemoryAllocator &allocator) {
    if (samples.size() % 2 != 0) {
        REPORT_FATAL_ERROR();
    }

    const int n = samples.size() / 2;
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

    init(cpu_lambdas, cpu_values, allocator);

    if (normalize) {
        scale(CIE_Y_integral / inner_product(cie_y), allocator);
    }
}

PiecewiseLinearSpectrum ::PiecewiseLinearSpectrum(const std::vector<Real> &_lambdas,
                                                  const std::vector<Real> &_values,
                                                  GPUMemoryAllocator &allocator) {
    init(_lambdas, _values, allocator);
}

void PiecewiseLinearSpectrum::init(const std::vector<Real> &_lambdas,
                                   const std::vector<Real> &_values,
                                   GPUMemoryAllocator &allocator) {
    if (_lambdas.size() != _values.size()) {
        REPORT_FATAL_ERROR();
    }
    size = _lambdas.size();

    auto gpu_lambdas = allocator.allocate<Real>(size);
    auto gpu_values = allocator.allocate<Real>(size);

    CHECK_CUDA_ERROR(
        cudaMemcpy(gpu_lambdas, _lambdas.data(), sizeof(Real) * size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        cudaMemcpy(gpu_values, _values.data(), sizeof(Real) * size, cudaMemcpyHostToDevice));

    lambdas = gpu_lambdas;
    values = gpu_values;
}

void PiecewiseLinearSpectrum::scale(const Real factor, GPUMemoryAllocator &allocator) {
    if (factor <= 0 || std::isnan(factor) || std::isinf(factor)) {
        REPORT_FATAL_ERROR();
    }

    auto scaled_values = allocator.allocate<Real>(size);
    for (int idx = 0; idx < size; ++idx) {
        scaled_values[idx] = values[idx] * factor;
    }

    allocator.release(values);

    values = scaled_values;
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
Real PiecewiseLinearSpectrum::operator()(const Real lambda) const {
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
