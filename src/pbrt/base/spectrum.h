#pragma once

#include "pbrt/spectrum_util/sampled_wavelengths.h"
#include "pbrt/spectrum_util/xyz.h"

class DenselySampledSpectrum;
class ConstantSpectrum;
class RGBIlluminantSpectrum;
class RGBAlbedoSpectrum;

class Spectrum {
  public:
    PBRT_CPU_GPU
    void init(const DenselySampledSpectrum *densely_sampled_spectrum);

    PBRT_CPU_GPU
    void init(const ConstantSpectrum *constant_spectrum);

    PBRT_CPU_GPU
    void init(const RGBIlluminantSpectrum *rgb_illuminant_spectrum);

    PBRT_CPU_GPU
    void init(const RGBAlbedoSpectrum *rgb_albedo_spectrum);

    PBRT_CPU_GPU FloatType operator()(FloatType lambda) const;

    PBRT_CPU_GPU
    FloatType inner_product(const Spectrum &spectrum) const {
        FloatType sum = 0;
        for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
            sum += (*this)(lambda)*spectrum(lambda);
        }

        return sum;
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    XYZ to_xyz(const Spectrum *cie_xyz[3]) const {
        auto x = cie_xyz[0];
        auto y = cie_xyz[1];
        auto z = cie_xyz[2];

        return XYZ(inner_product(*x), inner_product(*y), inner_product(*z)) / CIE_Y_integral;
    }

    PBRT_CPU_GPU
    FloatType to_photometric(const Spectrum &cie_y) const {
        return inner_product(cie_y);
    }

  private:
    enum class SpectrumType {
        densely_sampled_spectrum,
        constant_spectrum,
        rgb_illuminant_spectrum,
        rgb_albedo_spectrum,
    };

    SpectrumType spectrum_type;
    void *spectrum_ptr;
};
