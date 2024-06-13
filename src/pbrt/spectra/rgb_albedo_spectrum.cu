#include "pbrt/spectra/rgb_albedo_spectrum.h"

#include "pbrt/spectrum_util/rgb_color_space.h"

PBRT_CPU_GPU
void RGBAlbedoSpectrum::init(const RGB &rgb, const RGBColorSpace *cs) {
    rsp = cs->to_rgb_coefficients(rgb);
}

PBRT_CPU_GPU
SampledSpectrum RGBAlbedoSpectrum::sample(const SampledWavelengths &lambda) const {
    SampledSpectrum result;
    for (int i = 0; i < NSpectrumSamples; ++i) {
        result[i] = rsp(lambda[i]);
    }

    return result;
}
