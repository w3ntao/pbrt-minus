#pragma once

#include <pbrt/gpu/macro.h>

class GPUMemoryAllocator;
class SampledSpectrum;
class SampledWavelengths;
class Spectrum;

class PiecewiseLinearSpectrum {
  public:
    PiecewiseLinearSpectrum(const std::vector<Real> &samples, bool normalize, const Spectrum *cie_y,
                            GPUMemoryAllocator &allocator);

    PiecewiseLinearSpectrum(const std::vector<Real> &_lambdas, const std::vector<Real> &_values,
                            GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Real operator()(Real lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const;

    void scale(Real factor, GPUMemoryAllocator &allocator);

  private:
    const Real *lambdas = nullptr;
    const Real *values = nullptr;
    int size = 0;

    PBRT_CPU_GPU
    Real inner_product(const Spectrum *spectrum) const;

    void init(const std::vector<Real> &_lambdas, const std::vector<Real> &_values,
              GPUMemoryAllocator &allocator);
};
