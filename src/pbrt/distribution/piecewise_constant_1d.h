#pragma once

#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/gpu/macro.h>
#include <pbrt/util/math.h>
#include <vector>

class PiecewiseConstant1D {
  public:
    static const PiecewiseConstant1D *create(const std::vector<Real> &f, Real min, Real max,
                                             GPUMemoryAllocator &allocator) {
        return allocator.create<PiecewiseConstant1D>(f.data(), f.size(), min, max, allocator);
    }

    PiecewiseConstant1D() = default;

    PiecewiseConstant1D(const Real *f, const int f_size, const Real _min, const Real _max,
                        GPUMemoryAllocator &allocator)
        : size(f_size), min(_min), max(_max) {

        func = allocator.allocate<Real>(f_size);

        for (int idx = 0; idx < f_size; ++idx) {
            func[idx] = std::abs(f[idx]);
        }

        cdf = allocator.allocate<Real>(f_size + 1);

        cdf[0] = 0;
        size_t n = f_size;
        for (size_t i = 1; i < n + 1; ++i) {
            cdf[i] = cdf[i - 1] + func[i - 1] * (max - min) / n;
        }

        // Transform step function integral into CDF
        funcInt = cdf[n];
        if (funcInt == 0) {
            for (size_t i = 1; i < n + 1; ++i) {
                cdf[i] = Real(i) / Real(n);
            }
        } else {
            for (size_t i = 1; i < n + 1; ++i) {
                cdf[i] /= funcInt;
            }
        }
    }

    PBRT_CPU_GPU
    Real integral() const {
        return funcInt;
    }

    PBRT_CPU_GPU
    Real sample(Real u, Real *pdf = nullptr, int *offset = nullptr) const {
        // Find surrounding CDF segments and _offset_
        int o = find_interval(size + 1, [&](int index) { return cdf[index] <= u; });

        if (offset) {
            *offset = o;
        }

        // Compute offset along CDF segment
        Real du = u - cdf[o];

        if (cdf[o + 1] - cdf[o] > 0) {
            du /= cdf[o + 1] - cdf[o];
        }

        // Compute PDF for sampled offset
        if (pdf) {
            *pdf = (funcInt > 0) ? func[o] / funcInt : 0;
        }

        // Return $x$ corresponding to sample
        return pbrt::lerp((o + du) / size, min, max);
    }

  private:
    int size = 0;

    Real *func = nullptr; // size = _size
    Real *cdf = nullptr;  // size = _size + 1

    Real min = NAN;
    Real max = NAN;
    Real funcInt = NAN;
};
