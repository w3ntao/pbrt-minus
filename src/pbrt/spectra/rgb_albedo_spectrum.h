#pragma once

#include "pbrt/spectra/rgb_sigmoid_polynomial.h"

// TODO: rewrite RGBAlbedoSpectrum

class RGBAlbedoSpectrum {
  public:
    /*
    PBRT_GPU
    RGBAlbedoSpectrum(const RGBColorSpace &cs, const RGB &rgb) : rsp(cs.to_rgb_coefficients(rgb)) {}

    PBRT_GPU
    static std::array<RGBAlbedoSpectrum, 3> build_albedo_rgb(const RGBColorSpace &cs) {
        double val = 0.01;
        auto r = RGBAlbedoSpectrum(cs, RGB(val, 0.0, 0.0));
        auto g = RGBAlbedoSpectrum(cs, RGB(0.0, val, 0.0));
        auto b = RGBAlbedoSpectrum(cs, RGB(0.0, 0.0, val));

        return {r, g, b};
    }
    */

    PBRT_GPU double operator()(double lambda) const {
        return rsp(lambda);
    }

    PBRT_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const {
        double values[NSpectrumSamples];

        for (int i = 0; i < NSpectrumSamples; ++i) {
            values[i] = rsp(lambda[i]);
        }

        return SampledSpectrum(values);
    }

    PBRT_GPU
    void debug() const {
        printf("rsp values: ");
        rsp.debug();
    }

  private:
    RGBSigmoidPolynomial rsp;
};
