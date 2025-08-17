#pragma once

#include <pbrt/gpu/macro.h>
#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/spectrum_util/rgb_sigmoid_polynomial.h>

class RGBColorSpace;
class SampledSpectrum;
class SampledWavelengths;

class RGBUnboundedSpectrum {
  public:
    PBRT_CPU_GPU
    RGBUnboundedSpectrum(const RGB &rgb, const RGBColorSpace *cs);

    PBRT_CPU_GPU
    Real operator()(Real lambda) const {
        return scale * rsp(lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const;

  private:
    Real scale = NAN;
    RGBSigmoidPolynomial rsp;
};
