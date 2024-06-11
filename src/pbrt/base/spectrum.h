#pragma once

#include <vector>

#include "pbrt/spectrum_util/sampled_wavelengths.h"
#include "pbrt/spectrum_util/xyz.h"

class RGB;
class RGBColorSpace;

class ConstantSpectrum;
class DenselySampledSpectrum;
class PiecewiseLinearSpectrum;
class RGBIlluminantSpectrum;
class RGBAlbedoSpectrum;

enum class SpectrumType {
    Illuminant,
    Albedo,
    Unbounded,
};

class Spectrum {
  public:
    enum class Type {
        densely_sampled_spectrum,
        constant_spectrum,
        rgb_illuminant_spectrum,
        rgb_albedo_spectrum,
        piecewise_linear_spectrum,
    };

    static const Spectrum *create_cie_d(FloatType temperature, const FloatType *cie_s0,
                                        const FloatType *cie_s1, const FloatType *cie_s2,
                                        const FloatType *cie_lambda,
                                        std::vector<void *> &gpu_dynamic_pointer);

    static const Spectrum *create_constant_spectrum(FloatType val,
                                                    std::vector<void *> &gpu_dynamic_pointers);

    static const Spectrum *create_rgb_albedo_spectrum(const RGB &val,
                                                      std::vector<void *> &gpu_dynamic_pointers,
                                                      const RGBColorSpace *color_space);

    static const Spectrum *create_piecewise_linear_spectrum_from_lambdas_and_values(
        const std::vector<FloatType> &cpu_lambdas, const std::vector<FloatType> &cpu_values,
        std::vector<void *> &gpu_dynamic_pointers);

    static const Spectrum *
    create_piecewise_linear_spectrum_from_interleaved(const std::vector<FloatType> &samples,
                                                      bool normalize, const Spectrum *cie_y,
                                                      std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    void init(const DenselySampledSpectrum *densely_sampled_spectrum);

    PBRT_CPU_GPU
    void init(const ConstantSpectrum *constant_spectrum);

    PBRT_CPU_GPU
    void init(const RGBIlluminantSpectrum *rgb_illuminant_spectrum);

    PBRT_CPU_GPU
    void init(const RGBAlbedoSpectrum *rgb_albedo_spectrum);

    PBRT_CPU_GPU
    void init(const PiecewiseLinearSpectrum *piecewise_linear_spectrum);

    PBRT_CPU_GPU bool is_constant_spectrum() const {
        return type == Type::constant_spectrum;
    }

    PBRT_CPU_GPU FloatType operator()(FloatType lambda) const;

    PBRT_CPU_GPU
    FloatType inner_product(const Spectrum *spectrum) const {
        FloatType sum = 0;
        for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
            sum += (*this)(lambda) * (*spectrum)(lambda);
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

        return XYZ(inner_product(x), inner_product(y), inner_product(z)) / CIE_Y_integral;
    }

    PBRT_CPU_GPU
    FloatType to_photometric(const Spectrum *cie_y) const;

  private:
    Type type;
    const void *ptr;
};
