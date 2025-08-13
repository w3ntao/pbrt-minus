#pragma once

#include <pbrt/spectrum_util/sampled_wavelengths.h>
#include <pbrt/spectrum_util/xyz.h>

class RGB;
class RGBColorSpace;

class BlackbodySpectrum;
class ConstantSpectrum;
class DenselySampledSpectrum;
class GPUMemoryAllocator;
class RGBAlbedoSpectrum;
class RGBIlluminantSpectrum;
class RGBUnboundedSpectrum;
class PiecewiseLinearSpectrum;

enum class SpectrumType {
    Illuminant,
    Albedo,
    Unbounded,
};

class Spectrum {
  public:
    enum class Type {
        black_body,
        constant,
        densely_sampled,
        rgb_albedo,
        rgb_illuminant,
        rgb_unbounded,
        piecewise_linear,
    };

    static const Spectrum *create_black_body(Real temperature, GPUMemoryAllocator &allocator);

    static const Spectrum *create_cie_d(Real temperature, const Real *cie_s0, const Real *cie_s1,
                                        const Real *cie_s2, const Real *cie_lambda,
                                        GPUMemoryAllocator &allocator);

    static const Spectrum *create_constant_spectrum(Real val, GPUMemoryAllocator &allocator);

    static const Spectrum *create_from_rgb(const RGB &val, SpectrumType spectrum_type,
                                           const RGBColorSpace *color_space,
                                           GPUMemoryAllocator &allocator);

    static const Spectrum *
    create_piecewise_linear_spectrum_from_lambdas_and_values(const std::vector<Real> &cpu_lambdas,
                                                             const std::vector<Real> &cpu_values,
                                                             GPUMemoryAllocator &allocator);

    static const Spectrum *
    create_piecewise_linear_spectrum_from_interleaved(const std::vector<Real> &samples,
                                                      bool normalize, const Spectrum *cie_y,
                                                      GPUMemoryAllocator &allocator);
    PBRT_CPU_GPU
    void init(const BlackbodySpectrum *black_body_spectrum);

    PBRT_CPU_GPU
    void init(const ConstantSpectrum *constant_spectrum);

    PBRT_CPU_GPU
    void init(const DenselySampledSpectrum *densely_sampled_spectrum);

    PBRT_CPU_GPU
    void init(const RGBAlbedoSpectrum *rgb_albedo_spectrum);

    PBRT_CPU_GPU
    void init(const RGBIlluminantSpectrum *rgb_illuminant_spectrum);

    PBRT_CPU_GPU
    void init(const RGBUnboundedSpectrum *rgb_unbounded_spectrum);

    PBRT_CPU_GPU
    void init(const PiecewiseLinearSpectrum *piecewise_linear_spectrum);

    PBRT_CPU_GPU bool is_constant_spectrum() const {
        return type == Type::constant;
    }

    PBRT_CPU_GPU
    Real max_value() const;

    PBRT_CPU_GPU
    Real operator()(Real lambda) const;

    PBRT_CPU_GPU
    Real inner_product(const Spectrum *const spectrum) const {
        Real sum = 0;
        for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
            sum += (*this)(lambda) * (*spectrum)(lambda);
        }

        return sum;
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    XYZ to_xyz(const Spectrum *const cie_xyz[3]) const {
        return XYZ(inner_product(cie_xyz[0]), inner_product(cie_xyz[1]),
                   inner_product(cie_xyz[2])) /
               CIE_Y_integral;
    }

    PBRT_CPU_GPU
    Real to_photometric(const Spectrum *cie_y) const;

  private:
    Type type;
    const void *ptr;
};
