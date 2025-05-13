#pragma once

#include <pbrt/gpu/macro.h>

class GPUMemoryAllocator;
class SampledSpectrum;
class SampledWavelengths;
class Spectrum;

class PiecewiseLinearSpectrum {
  public:
    static const PiecewiseLinearSpectrum *
    create_from_lambdas_values(const std::vector<Real> &cpu_lambdas,
                               const std::vector<Real> &cpu_values,
                               GPUMemoryAllocator &allocator);

    static const PiecewiseLinearSpectrum *
    create_from_interleaved(const std::vector<Real> &samples, bool normalize,
                            const Spectrum *cie_y, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Real operator()(Real lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    void scale(Real scaling_factor) {
        for (uint idx = 0; idx < size; ++idx) {
            values[idx] = values[idx] * scaling_factor;
        }
    }

  private:
    Real *lambdas;
    Real *values;
    uint size;

    PBRT_CPU_GPU
    Real inner_product(const Spectrum *spectrum) const;

    void init_from_lambdas_values(const std::vector<Real> &cpu_lambdas,
                                  const std::vector<Real> &cpu_values,
                                  GPUMemoryAllocator &allocator);

    void init_from_interleaved(const std::vector<Real> &samples, bool normalize,
                               const Spectrum *cie_y, GPUMemoryAllocator &allocator);
};
