#pragma once

#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/spectrum_util/rgb_sigmoid_polynomial.h>
#include <pbrt/spectrum_util/sampled_spectrum.h>

class RGBColorSpace;
class SampledWavelengths;

class RGBAlbedoSpectrum {
  public:
    PBRT_CPU_GPU
    void init(const RGB &rgb, const RGBColorSpace *cs);

    PBRT_CPU_GPU Real operator()(Real lambda) const {
        return rsp(lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const;

  private:
    RGBSigmoidPolynomial rsp;
};
