#include "pbrt/spectra/densely_sampled_spectrum.h"
#include "pbrt/base/spectrum.h"

PBRT_CPU_GPU
FloatType DenselySampledSpectrum::inner_product(const Spectrum *spectrum) const {
    FloatType sum = 0;
    for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
        sum += (*this)(lambda) * (*spectrum)(lambda);
    }

    return sum;
}

PBRT_CPU_GPU
void DenselySampledSpectrum::init_from_spectrum(const Spectrum *spectrum) {
    for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
        values[lambda - LAMBDA_MIN] = (*spectrum)(lambda);
    }
}

PBRT_CPU_GPU
void DenselySampledSpectrum::init_from_pls_interleaved_samples(const FloatType *samples,
                                                               uint num_samples, bool normalize,
                                                               const Spectrum *cie_y) {
    if (num_samples % 2 != 0 || num_samples / 2 + 2 > LAMBDA_RANGE) {
        printf("DenselySampledSpectrum::init_from_pls_interleaved_samples(): "
               "illegal num_samples");
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::runtime_error("DenselySampledSpectrum::init_from_pls_interleaved_samples()");
#endif
    }

    FloatType _lambdas[LAMBDA_RANGE];
    FloatType _values[LAMBDA_RANGE];

    uint offset = 0;

    // Extend samples to cover range of visible wavelengths if needed.
    if (samples[0] > LAMBDA_MIN) {
        _lambdas[0] = LAMBDA_MIN - 1;
        _values[0] = samples[1];
        offset += 1;
    }

    for (uint idx = 0; idx < num_samples / 2; ++idx) {
        _lambdas[idx + offset] = samples[idx * 2];
        _values[idx + offset] = samples[idx * 2 + 1];
    }

    // Extend samples to cover range of visible wavelengths if needed.
    if (samples[num_samples - 2] < LAMBDA_MAX) {
        _lambdas[num_samples / 2 + offset] = LAMBDA_MAX + 1;
        _values[num_samples / 2 + offset] = samples[num_samples - 1];
        offset += 1;
    }

    init_from_pls_lambdas_values(_lambdas, _values, num_samples / 2 + offset);

    if (normalize) {
        scale(CIE_Y_integral / inner_product(cie_y));
    }
}
PBRT_CPU_GPU
void DenselySampledSpectrum::init_cie_d(FloatType temperature, const FloatType *cie_s0,
                                        const FloatType *cie_s1, const FloatType *cie_s2,
                                        const FloatType *cie_lambda) {
    FloatType cct = temperature * 1.4388f / 1.4380f;
    if (cct < 4000) {
        // CIE D ill-defined, use blackbody
        BlackbodySpectrum bb = BlackbodySpectrum(cct);
        init_with_sample_function([=](FloatType lambda) { return bb(lambda); });
        return;
    }

    // Convert CCT to xy
    FloatType x = cct <= 7000 ? -4.607f * 1e9f / std::pow(cct, 3) + 2.9678f * 1e6f / sqr(cct) +
                                    0.09911f * 1e3f / cct + 0.244063f
                              : -2.0064f * 1e9f / std::pow(cct, 3) + 1.9018f * 1e6f / sqr(cct) +
                                    0.24748f * 1e3f / cct + 0.23704f;

    FloatType y = -3 * x * x + 2.870f * x - 0.275f;

    // Interpolate D spectrum
    FloatType M = 0.0241f + 0.2562f * x - 0.7341f * y;
    FloatType M1 = (-1.3515f - 1.7703f * x + 5.9114f * y) / M;
    FloatType M2 = (0.0300f - 31.4424f * x + 30.0717f * y) / M;

    FloatType _values[nCIES];
    for (int i = 0; i < nCIES; ++i) {
        _values[i] = (cie_s0[i] + cie_s1[i] * M1 + cie_s2[i] * M2) * 0.01;
    }

    init_from_pls_lambdas_values(cie_lambda, _values, nCIES);
}