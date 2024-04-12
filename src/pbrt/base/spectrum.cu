#include "pbrt/base/spectrum.h"
#include "pbrt/spectra/densely_sampled_spectrum.h"

PBRT_CPU_GPU
void Spectrum::init(const DenselySampledSpectrum *densely_sampled_spectrum) {
    spectrum_type = SpectrumType::densely_sampled;
    spectrum_ptr = (void *)densely_sampled_spectrum;
}

PBRT_CPU_GPU
FloatType Spectrum::operator()(FloatType lambda) const {
    switch (spectrum_type) {
    case (SpectrumType::densely_sampled): {
        return ((DenselySampledSpectrum *)spectrum_ptr)->operator()(lambda);
    }
    }

    printf("\nSpectrum::operator(): not implemented for this type\n\n");
#if defined(__CUDA_ARCH__)
    asm("trap;");
#else
    throw std::runtime_error("Spectrum::operator(): not implemented for this type\n");
#endif
}

PBRT_CPU_GPU
SampledSpectrum Spectrum::sample(const SampledWavelengths &lambda) const {
    switch (spectrum_type) {
    case (SpectrumType::densely_sampled): {
        return ((DenselySampledSpectrum *)spectrum_ptr)->sample(lambda);
    }
    }

    printf("\nSpectrum::operator(): not implemented for this type\n\n");
#if defined(__CUDA_ARCH__)
    asm("trap;");
#else
    throw std::runtime_error("Spectrum::operator(): not implemented for this type\n");
#endif
}