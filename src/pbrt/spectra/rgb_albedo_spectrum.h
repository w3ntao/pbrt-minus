#pragma once

#include <pbrt/spectrum_util/rgb_color_space.h>
#include <pbrt/spectrum_util/rgb_sigmoid_polynomial.h>

class RGBAlbedoSpectrum {
  public:
    PBRT_CPU_GPU
    RGBAlbedoSpectrum() {}

    PBRT_CPU_GPU
    RGBAlbedoSpectrum(const RGB &rgb, const RGBColorSpace *cs)
        : rsp(cs->to_rgb_coefficients(rgb)) {}

    PBRT_CPU_GPU
    Real operator()(const Real lambda) const {
        return rsp(lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const {
        SampledSpectrum result;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            result[i] = rsp(lambda[i]);
        }

        return result;
    }

  private:
    RGBSigmoidPolynomial rsp;
};
