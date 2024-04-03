#include "pbrt/spectra/densely_sampled_spectrum.h"
#include "pbrt/base/spectrum.h"

PBRT_CPU_GPU
double DenselySampledSpectrum::inner_product(const Spectrum &spectrum) const {
    double sum = 0;
    for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
        sum += (*this)(lambda)*spectrum(lambda);
    }

    return sum;
}

PBRT_CPU_GPU void DenselySampledSpectrum::init_from_pls_interleaved_samples(const double *samples,
                                                                            uint num_samples,
                                                                            bool normalize,
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

    double _lambdas[LAMBDA_RANGE];
    double _values[LAMBDA_RANGE];

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
        scale(CIE_Y_integral / inner_product(*cie_y));
    }
}
