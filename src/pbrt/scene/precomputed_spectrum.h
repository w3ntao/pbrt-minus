#pragma once

#include <map>
#include <vector>

#include "pbrt/util/macro.h"

class Spectrum;
class ThreadPool;

namespace RGBtoSpectrumData {
class RGBtoSpectrumTable;
}

struct PreComputedSpectrum {
    explicit PreComputedSpectrum(ThreadPool &thread_pool);

    ~PreComputedSpectrum() {
        for (auto ptr : gpu_dynamic_pointers) {
            CHECK_CUDA_ERROR(cudaFree(ptr));
        }

        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    const Spectrum *cie_xyz[3] = {nullptr};
    const Spectrum *illum_d65 = nullptr;

    RGBtoSpectrumData::RGBtoSpectrumTable *rgb_to_spectrum_table = nullptr;

    std::map<std::string, const Spectrum *> named_spectra;

  private:
    std::vector<void *> gpu_dynamic_pointers;
};
