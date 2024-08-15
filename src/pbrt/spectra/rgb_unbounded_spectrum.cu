#include "pbrt/spectra/rgb_unbounded_spectrum.h"

#include "pbrt/spectrum_util/rgb_color_space.h"

PBRT_CPU_GPU
RGBUnboundedSpectrum::RGBUnboundedSpectrum(RGB rgb, const RGBColorSpace *cs) {
    this->init(rgb, cs);
}

PBRT_CPU_GPU
void RGBUnboundedSpectrum::init(RGB rgb, const RGBColorSpace *cs) {
    auto m = std::max({rgb.r, rgb.g, rgb.b});
    scale = 2 * m;

    if (scale < 0) {
        REPORT_FATAL_ERROR();
    }

    rsp = cs->to_rgb_coefficients(scale > 0 ? rgb / scale : RGB(0, 0, 0));
}

PBRT_CPU_GPU
SampledSpectrum RGBUnboundedSpectrum::sample(const SampledWavelengths &lambda) const {
    SampledSpectrum s;
    for (int idx = 0; idx < NSpectrumSamples; ++idx) {
        s[idx] = scale * rsp(lambda[idx]);
    }

    return s;
}
