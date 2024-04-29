#pragma once

#include "pbrt/util/macro.h"
#include "pbrt/spectrum_util/rgb.h"
#include "pbrt/spectrum_util/rgb_sigmoid_polynomial.h"

class Spectrum;
class RGBColorSpace;

class RGBIlluminantSpectrum {
  public:
    void init(const RGB &rgb, const RGBColorSpace *rgb_color_space);

    PBRT_CPU_GPU
    FloatType operator()(FloatType lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    FloatType to_photometric(const Spectrum *cie_y) const {
        return inner_product(cie_y);
    }

  private:
    FloatType scale;
    RGBSigmoidPolynomial rsp;
    const Spectrum *illuminant;

    PBRT_CPU_GPU
    FloatType inner_product(const Spectrum *spectrum) const;
};
