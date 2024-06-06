#pragma once

#include "pbrt/util/thread_pool.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/spectrum_util/rgb_color_space.h"

struct PreComputedSpectrum {
    explicit PreComputedSpectrum(ThreadPool &thread_pool) {
        auto start = std::chrono::system_clock::now();

        for (uint idx = 0; idx < 3; idx++) {
            CHECK_CUDA_ERROR(
                cudaMallocManaged(&(dense_cie_xyz[idx]), sizeof(DenselySampledSpectrum)));
        }
        dense_cie_xyz[0]->init_from_pls_lambdas_values(CIE_LAMBDA_CPU, CIE_X_VALUE_CPU,
                                                       NUM_CIE_SAMPLES);
        dense_cie_xyz[1]->init_from_pls_lambdas_values(CIE_LAMBDA_CPU, CIE_Y_VALUE_CPU,
                                                       NUM_CIE_SAMPLES);
        dense_cie_xyz[2]->init_from_pls_lambdas_values(CIE_LAMBDA_CPU, CIE_Z_VALUE_CPU,
                                                       NUM_CIE_SAMPLES);

        for (uint idx = 0; idx < 3; idx++) {
            Spectrum *_spectrum;
            CHECK_CUDA_ERROR(cudaMallocManaged(&_spectrum, sizeof(Spectrum)));
            _spectrum->init(dense_cie_xyz[idx]);
            cie_xyz[idx] = _spectrum;
        }

        CHECK_CUDA_ERROR(cudaMallocManaged(&dense_illum_d65, sizeof(DenselySampledSpectrum)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&illum_d65, sizeof(Spectrum)));

        dense_illum_d65->init_from_pls_interleaved_samples(
            CIE_Illum_D6500, sizeof(CIE_Illum_D6500) / sizeof(CIE_Illum_D6500[0]), true,
            cie_xyz[1]);
        illum_d65->init(dense_illum_d65);

        CHECK_CUDA_ERROR(cudaMallocManaged(&rgb_to_spectrum_table,
                                           sizeof(RGBtoSpectrumData::RGBtoSpectrumTable)));
        rgb_to_spectrum_table->init("sRGB", thread_pool);

        const std::chrono::duration<FloatType> duration{std::chrono::system_clock::now() - start};
        std::cout << std::fixed << std::setprecision(1) << "spectra computing took "
                  << duration.count() << " seconds.\n"
                  << std::flush;
    }

    ~PreComputedSpectrum() {
        for (auto ptr : std::vector<void *>{
                 rgb_to_spectrum_table,
                 dense_illum_d65,
                 illum_d65,
             }) {
            CHECK_CUDA_ERROR(cudaFree(ptr));
        }

        for (uint idx = 0; idx < 3; idx++) {
            CHECK_CUDA_ERROR(cudaFree(dense_cie_xyz[idx]));
            CHECK_CUDA_ERROR(cudaFree((void *)cie_xyz[idx]));
        }

        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    const Spectrum *cie_xyz[3] = {nullptr};
    Spectrum *illum_d65 = nullptr;

    RGBtoSpectrumData::RGBtoSpectrumTable *rgb_to_spectrum_table = nullptr;

  private:
    DenselySampledSpectrum *dense_cie_xyz[3] = {nullptr};
    DenselySampledSpectrum *dense_illum_d65 = nullptr;
};
