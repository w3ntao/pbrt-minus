#include "pbrt/scene/precomputed_spectrum.h"

#include "pbrt/base/spectrum.h"
#include "pbrt/spectrum_util/rgb_color_space.h"
#include "pbrt/spectrum_util/spectrum_constants_metal.h"
#include "pbrt/util/thread_pool.h"

PreComputedSpectrum::PreComputedSpectrum(ThreadPool &thread_pool) {
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

    illum_d65 = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(CIE_Illum_D6500), std::end(CIE_Illum_D6500)), true, cie_xyz[1],
        gpu_dynamic_pointers);

    CHECK_CUDA_ERROR(
        cudaMallocManaged(&rgb_to_spectrum_table, sizeof(RGBtoSpectrumData::RGBtoSpectrumTable)));
    gpu_dynamic_pointers.push_back(rgb_to_spectrum_table);

    rgb_to_spectrum_table->init("sRGB", thread_pool);

    auto ag_eta = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(Ag_eta), std::end(Ag_eta)), false, nullptr, gpu_dynamic_pointers);

    auto ag_k = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(Ag_k), std::end(Ag_k)), false, nullptr, gpu_dynamic_pointers);

    named_spectra = {
        {"metal-Ag-eta", ag_eta},
        {"metal-Ag-k", ag_k},
    };
    
    const std::chrono::duration<FloatType> duration{std::chrono::system_clock::now() - start};
    std::cout << std::fixed << std::setprecision(1) << "spectra computing took " << duration.count()
              << " seconds.\n"
              << std::flush;
}