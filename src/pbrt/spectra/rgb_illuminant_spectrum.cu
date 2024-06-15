#include "pbrt/base/spectrum.h"
#include "pbrt/spectra/rgb_illuminant_spectrum.h"
#include "pbrt/spectrum_util/rgb_color_space.h"

PBRT_CPU_GPU
void RGBIlluminantSpectrum::init(const RGB &rgb, const RGBColorSpace *rgb_color_space) {
    illuminant = rgb_color_space->illuminant;
    scale = 2.0 * rgb.max_component();
    rsp = rgb_color_space->to_rgb_coefficients(scale > 0.0 ? rgb / scale : RGB(0.0, 0.0, 0.0));
}

PBRT_CPU_GPU
FloatType RGBIlluminantSpectrum::operator()(FloatType lambda) const {
    return scale * rsp(lambda) * (*illuminant)(lambda);
}

PBRT_CPU_GPU
FloatType RGBIlluminantSpectrum::to_photometric(const Spectrum *cie_y) const {
    return illuminant->inner_product(cie_y);
}

PBRT_CPU_GPU
SampledSpectrum RGBIlluminantSpectrum::sample(const SampledWavelengths &lambda) const {
    SampledSpectrum s;
    for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
        s[idx] = scale * rsp(lambda[idx]);
    }

    return illuminant->sample(lambda) * s;
}
