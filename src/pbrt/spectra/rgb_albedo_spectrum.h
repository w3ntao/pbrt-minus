#pragma once

#include "pbrt/spectra/rgb_sigmoid_polynomial.h"

class RGBAlbedoSpectrum {
  public:
    PBRT_CPU_GPU
    RGBAlbedoSpectrum(const RGBColorSpace *cs, const RGB &rgb)
        : rsp(cs->to_rgb_coefficients(rgb)) {}

    PBRT_CPU_GPU
    static void build_albedo_rgb(RGBAlbedoSpectrum out[3], const RGBColorSpace *cs) {
        double val = 0.01;
        out[0] = RGBAlbedoSpectrum(cs, RGB(val, 0.0, 0.0));
        out[1] = RGBAlbedoSpectrum(cs, RGB(0.0, val, 0.0));
        out[2] = RGBAlbedoSpectrum(cs, RGB(0.0, 0.0, val));
    }

    PBRT_CPU_GPU double operator()(double lambda) const {
        return rsp(lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const {
        double values[NSpectrumSamples];

        for (int i = 0; i < NSpectrumSamples; ++i) {
            values[i] = rsp(lambda[i]);
        }

        return SampledSpectrum(values);
    }

  private:
    RGBSigmoidPolynomial rsp;
};
