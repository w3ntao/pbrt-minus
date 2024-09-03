#pragma once

#include "pbrt/util/macro.h"
#include <vector>

class SampledSpectrum;
class SampledWavelengths;
class Spectrum;

class PiecewiseLinearSpectrum {
  public:
    static const PiecewiseLinearSpectrum *
    create_from_lambdas_values(const std::vector<FloatType> &cpu_lambdas,
                               const std::vector<FloatType> &cpu_values,
                               std::vector<void *> &gpu_dynamic_pointers);

    static const PiecewiseLinearSpectrum *
    create_from_interleaved(const std::vector<FloatType> &samples, bool normalize,
                            const Spectrum *cie_y, std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    FloatType operator()(FloatType lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    void scale(FloatType scaling_factor) {
        for (uint idx = 0; idx < size; ++idx) {
            values[idx] = values[idx] * scaling_factor;
        }
    }

  private:
    FloatType *lambdas;
    FloatType *values;
    uint size;

    PBRT_CPU_GPU
    FloatType inner_product(const Spectrum *spectrum) const;

    void init_from_lambdas_values(const std::vector<FloatType> &cpu_lambdas,
                                  const std::vector<FloatType> &cpu_values,
                                  std::vector<void *> &gpu_dynamic_pointers);

    void init_from_interleaved(const std::vector<FloatType> &samples, bool normalize,
                               const Spectrum *cie_y, std::vector<void *> &gpu_dynamic_pointers);
};
