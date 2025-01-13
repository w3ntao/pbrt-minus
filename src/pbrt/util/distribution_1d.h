#pragma once

#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/gpu/macro.h>
#include <cuda/std/tuple>
#include <functional>
#include <numeric>

PBRT_CPU_GPU
static uint search_cdf(FloatType u, const FloatType *cdf, uint length) {
    uint target_idx = UINT_MAX;

    if (u < cdf[0] || length == 1) {
        target_idx = 0;
    } else {
        uint start = 1;
        uint end = length;

        while (true) {
            if (end - start <= 10) {
                target_idx = start;
                for (auto idx = start; idx < end; ++idx) {
                    if (u >= cdf[idx - 1] && u < cdf[idx]) {
                        target_idx = idx;
                        break;
                    }
                }
                break;
            }

            auto mid = (start + end) / 2;
            if (u >= cdf[mid]) {
                // notice: change pivot to mid+1, rather than mid
                start = mid + 1;
            } else {
                end = mid + 1;
            }
        }
    }

    return target_idx;
}

class Distribution1D {
  public:
    void build(const std::vector<FloatType> &pdfs, GPUMemoryAllocator &allocator) {
        num = pdfs.size();

        auto _pmf = allocator.allocate<FloatType>(num);

        const double sum_pdf = std::accumulate(pdfs.begin(), pdfs.end(), 0.0);
        if (sum_pdf == 0.0) {
            for (uint idx = 0; idx < num; ++idx) {
                _pmf[idx] = 1.0 / num;
            }
        } else {
            for (uint idx = 0; idx < num; ++idx) {
                _pmf[idx] = pdfs[idx] / sum_pdf;
            }
        }

        auto _cdf = allocator.allocate<FloatType>(num);

        _cdf[0] = _pmf[0];
        for (size_t idx = 1; idx < num; ++idx) {
            _cdf[idx] = _cdf[idx - 1] + _pmf[idx];
        }

        cdf = _cdf;
        pdf = _pmf;
    }

    PBRT_CPU_GPU
    cuda::std::pair<uint, FloatType> sample(FloatType u) const {
        auto target_idx = search_cdf(u, cdf, num);
        return {target_idx, pdf[target_idx]};
    }

    PBRT_CPU_GPU FloatType get_pdf(uint idx) const {
        return pdf[idx];
    }

  private:
    const FloatType *cdf;
    const FloatType *pdf;
    uint num;
};
