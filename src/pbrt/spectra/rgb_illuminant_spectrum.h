#pragma once

#include <pbrt/gpu/macro.h>
#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/spectrum_util/rgb_sigmoid_polynomial.h>

class Spectrum;
class RGBColorSpace;

class RGBIlluminantSpectrum {
  public:
    PBRT_CPU_GPU
    RGBIlluminantSpectrum(const RGB &rgb, const RGBColorSpace *rgb_color_space) {
        init(rgb, rgb_color_space);
    }

    PBRT_CPU_GPU
    void init(const RGB &rgb, const RGBColorSpace *rgb_color_space);

    PBRT_CPU_GPU
    Real max_value() const;

    PBRT_CPU_GPU
    Real operator()(Real lambda) const;

    PBRT_CPU_GPU
    Real to_photometric(const Spectrum *cie_y) const;

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const;

  private:
    Real scale;
    RGBSigmoidPolynomial rsp;
    const Spectrum *illuminant;
};
