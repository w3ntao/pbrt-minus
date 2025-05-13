#pragma once

#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/spectrum_util/rgb_sigmoid_polynomial.h>
#include <pbrt/gpu/macro.h>

class RGBColorSpace;
class SampledSpectrum;
class SampledWavelengths;

class RGBUnboundedSpectrum {
  public:
    PBRT_CPU_GPU
    RGBUnboundedSpectrum() : rsp(0, 0, 0), scale(0) {}

    PBRT_CPU_GPU
    RGBUnboundedSpectrum(RGB rgb, const RGBColorSpace *cs);

    PBRT_CPU_GPU
    void init(RGB rgb, const RGBColorSpace *cs);

    PBRT_CPU_GPU
    Real operator()(Real lambda) const {
        return scale * rsp(lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const;

  private:
    Real scale;
    RGBSigmoidPolynomial rsp;
};
