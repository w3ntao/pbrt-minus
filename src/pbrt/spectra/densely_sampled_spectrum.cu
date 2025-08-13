#include <pbrt/base/spectrum.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectra/densely_sampled_spectrum.h>

const DenselySampledSpectrum *DenselySampledSpectrum::create(const Spectrum *spectrum,
                                                             const Real scale,
                                                             GPUMemoryAllocator &allocator) {
    auto dense_sampled_spectrum = allocator.allocate<DenselySampledSpectrum>();
    dense_sampled_spectrum->init_from_spectrum(spectrum, scale);

    return dense_sampled_spectrum;
}

PBRT_CPU_GPU
Real DenselySampledSpectrum::inner_product(const Spectrum *spectrum) const {
    Real sum = 0;
    for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
        sum += (*this)(lambda) * (*spectrum)(lambda);
    }

    return sum;
}

PBRT_CPU_GPU
void DenselySampledSpectrum::init_from_spectrum(const Spectrum *spectrum, const Real scale) {
    for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
        values[lambda - LAMBDA_MIN] = (*spectrum)(lambda)*scale;
    }
}
