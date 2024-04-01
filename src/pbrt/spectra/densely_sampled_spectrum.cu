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
    if (num_samples > 2 * LAMBDA_RANGE) {
        printf("DenselySampledSpectrum::init_from_pls_interleaved_samples(): num_samples too "
               "large to "
               "handle.");
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::runtime_error("DenselySampledSpectrum::init_from_pls_interleaved_samples()");
#endif
    }

    double _lambdas[LAMBDA_RANGE];
    double _values[LAMBDA_RANGE];

    for (uint i = 0; i < num_samples / 2; ++i) {
        _lambdas[i] = samples[i * 2];
        _values[i] = samples[i * 2 + 1];
    }

    init_from_pls_lambdas_values(_lambdas, _values, num_samples / 2);

    if (normalize) {
        scale(CIE_Y_integral / inner_product(*cie_y));
    }
}
