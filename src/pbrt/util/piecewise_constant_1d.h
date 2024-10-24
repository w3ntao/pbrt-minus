#pragma once

#include "pbrt/util/macro.h"
#include <vector>

class PiecewiseConstant1D {
  public:
    static const PiecewiseConstant1D *create(const std::vector<FloatType> &f, FloatType min,
                                             FloatType max,
                                             std::vector<void *> &gpu_dynamic_pointers) {
        PiecewiseConstant1D *piecewise_constant_1D;
        CHECK_CUDA_ERROR(cudaMallocManaged(&piecewise_constant_1D, sizeof(PiecewiseConstant1D)));
        gpu_dynamic_pointers.push_back(piecewise_constant_1D);

        piecewise_constant_1D->init(f.data(), f.size(), min, max, gpu_dynamic_pointers);

        return piecewise_constant_1D;
    }

    void init(const FloatType *f, uint f_size, FloatType _min, FloatType _max,
              std::vector<void *> &gpu_dynamic_pointers) {
        cdf = nullptr;
        func = nullptr;

        min = _min;
        max = _max;
        _size = f_size;

        CHECK_CUDA_ERROR(cudaMallocManaged(&func, sizeof(FloatType) * f_size));
        gpu_dynamic_pointers.push_back(func);

        for (uint idx = 0; idx < f_size; ++idx) {
            func[idx] = std::abs(f[idx]);
        }

        CHECK_CUDA_ERROR(cudaMallocManaged(&cdf, sizeof(FloatType) * (f_size + 1)));
        gpu_dynamic_pointers.push_back(cdf);

        cdf[0] = 0;
        size_t n = f_size;
        for (size_t i = 1; i < n + 1; ++i) {
            cdf[i] = cdf[i - 1] + func[i - 1] * (max - min) / n;
        }

        // Transform step function integral into CDF
        funcInt = cdf[n];
        if (funcInt == 0) {
            for (size_t i = 1; i < n + 1; ++i) {
                cdf[i] = FloatType(i) / FloatType(n);
            }
        } else {
            for (size_t i = 1; i < n + 1; ++i) {
                cdf[i] /= funcInt;
            }
        }
    }

    PBRT_CPU_GPU
    FloatType integral() const {
        return funcInt;
    }

    PBRT_CPU_GPU uint size() const {
        return _size;
    }

    PBRT_CPU_GPU
    FloatType sample(FloatType u, FloatType *pdf = nullptr, int *offset = nullptr) const {
        // Find surrounding CDF segments and _offset_
        int o = find_interval(_size + 1, [&](int index) { return cdf[index] <= u; });

        if (offset) {
            *offset = o;
        }

        // Compute offset along CDF segment
        FloatType du = u - cdf[o];

        if (cdf[o + 1] - cdf[o] > 0) {
            du /= cdf[o + 1] - cdf[o];
        }

        // Compute PDF for sampled offset
        if (pdf) {
            *pdf = (funcInt > 0) ? func[o] / funcInt : 0;
        }

        // Return $x$ corresponding to sample
        return lerp((o + du) / size(), min, max);
    }

  private:
    uint _size;

    FloatType *func; // size = _size
    FloatType *cdf;  // size = _size + 1

    FloatType min;
    FloatType max;
    FloatType funcInt;
};
