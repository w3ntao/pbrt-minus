#pragma once

#include "pbrt/util/thread_pool.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/spectrum_util/rgb_color_space.h"

struct PreComputedSpectrum {
    explicit PreComputedSpectrum(ThreadPool &thread_pool) {
        auto start = std::chrono::system_clock::now();

        std::vector<FloatType> cpu_cie_lambdas(NUM_CIE_SAMPLES);
        std::vector<FloatType> cpu_cie_x_values(NUM_CIE_SAMPLES);
        std::vector<FloatType> cpu_cie_y_values(NUM_CIE_SAMPLES);
        std::vector<FloatType> cpu_cie_z_values(NUM_CIE_SAMPLES);

        for (uint idx = 0; idx < NUM_CIE_SAMPLES; ++idx) {
            cpu_cie_lambdas.push_back(CIE_LAMBDA_CPU[idx]);
            cpu_cie_x_values.push_back(CIE_X_VALUE_CPU[idx]);
            cpu_cie_y_values.push_back(CIE_Y_VALUE_CPU[idx]);
            cpu_cie_z_values.push_back(CIE_Z_VALUE_CPU[idx]);
        }

        cie_xyz[0] = Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
            cpu_cie_lambdas, cpu_cie_x_values, gpu_dynamic_pointers);
        cie_xyz[1] = Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
            cpu_cie_lambdas, cpu_cie_y_values, gpu_dynamic_pointers);
        cie_xyz[2] = Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
            cpu_cie_lambdas, cpu_cie_z_values, gpu_dynamic_pointers);

        const uint samples_size = sizeof(CIE_Illum_D6500) / sizeof(CIE_Illum_D6500[0]);
        std::vector<FloatType> cie_illum_d6500_samples(samples_size);
        for (uint idx = 0; idx < samples_size; ++idx) {
            cie_illum_d6500_samples.push_back(CIE_Illum_D6500[idx]);
        }

        illum_d65 = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
            cie_illum_d6500_samples, true, cie_xyz[1], gpu_dynamic_pointers);

        CHECK_CUDA_ERROR(cudaMallocManaged(&rgb_to_spectrum_table,
                                           sizeof(RGBtoSpectrumData::RGBtoSpectrumTable)));
        rgb_to_spectrum_table->init("sRGB", thread_pool);

        gpu_dynamic_pointers.push_back(rgb_to_spectrum_table);

        const std::chrono::duration<FloatType> duration{std::chrono::system_clock::now() - start};
        std::cout << std::fixed << std::setprecision(1) << "spectra computing took "
                  << duration.count() << " seconds.\n"
                  << std::flush;
    }

    ~PreComputedSpectrum() {
        for (auto ptr : gpu_dynamic_pointers) {
            CHECK_CUDA_ERROR(cudaFree(ptr));
        }

        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    const Spectrum *cie_xyz[3] = {nullptr};
    const Spectrum *illum_d65 = nullptr;

    RGBtoSpectrumData::RGBtoSpectrumTable *rgb_to_spectrum_table = nullptr;

  private:
    std::vector<void *> gpu_dynamic_pointers;
};
