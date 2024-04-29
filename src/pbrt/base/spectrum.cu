#include "pbrt/base/spectrum.h"

#include "pbrt/spectra/densely_sampled_spectrum.h"
#include "pbrt/spectra/const_spectrum.h"
#include "pbrt/spectra/rgb_illuminant_spectrum.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"

PBRT_CPU_GPU
void Spectrum::init(const DenselySampledSpectrum *densely_sampled_spectrum) {
    spectrum_type = SpectrumType::densely_sampled_spectrum;
    spectrum_ptr = (void *)densely_sampled_spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const ConstantSpectrum *constant_spectrum) {
    spectrum_type = SpectrumType::constant_spectrum;
    spectrum_ptr = (void *)constant_spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const RGBIlluminantSpectrum *rgb_illuminant_spectrum) {
    spectrum_type = SpectrumType::rgb_illuminant_spectrum;
    spectrum_ptr = (void *)rgb_illuminant_spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const RGBAlbedoSpectrum *rgb_albedo_spectrum) {
    spectrum_type = SpectrumType::rgb_albedo_spectrum;
    spectrum_ptr = (void *)rgb_albedo_spectrum;
}

PBRT_CPU_GPU
FloatType Spectrum::operator()(FloatType lambda) const {
    switch (spectrum_type) {
    case (SpectrumType::densely_sampled_spectrum): {
        return ((DenselySampledSpectrum *)spectrum_ptr)->operator()(lambda);
    }

    case (SpectrumType::constant_spectrum): {
        return ((ConstantSpectrum *)spectrum_ptr)->operator()(lambda);
    }

    case (SpectrumType::rgb_illuminant_spectrum): {
        return ((RGBIlluminantSpectrum *)spectrum_ptr)->operator()(lambda);
    }

    case (SpectrumType::rgb_albedo_spectrum): {
        return ((RGBAlbedoSpectrum *)spectrum_ptr)->operator()(lambda);
    }
    }

    report_function_error_and_exit(__func__);
    return NAN;
}

PBRT_CPU_GPU
SampledSpectrum Spectrum::sample(const SampledWavelengths &lambda) const {
    switch (spectrum_type) {
    case (SpectrumType::densely_sampled_spectrum): {
        return ((DenselySampledSpectrum *)spectrum_ptr)->sample(lambda);
    }

    case (SpectrumType::constant_spectrum): {
        return ((ConstantSpectrum *)spectrum_ptr)->sample(lambda);
    }

    case (SpectrumType::rgb_illuminant_spectrum): {
        return ((RGBIlluminantSpectrum *)spectrum_ptr)->sample(lambda);
    }

    case (SpectrumType::rgb_albedo_spectrum): {
        return ((RGBAlbedoSpectrum *)spectrum_ptr)->sample(lambda);
    }
    }

    report_function_error_and_exit(__func__);
    return SampledSpectrum::same_value(NAN);
}
