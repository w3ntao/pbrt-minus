#include "pbrt/base/spectrum.h"
#include "pbrt/spectrum_util/global_spectra.h"
#include "pbrt/spectrum_util/rgb_color_space.h"
#include "pbrt/util/thread_pool.h"

const GlobalSpectra *GlobalSpectra::create(RGBtoSpectrumData::Gamut gamut, ThreadPool &thread_pool,
                                           std::vector<void *> &gpu_dynamic_pointers) {
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

    const Spectrum *cie_xyz[3];
    cie_xyz[0] = Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
        cpu_cie_lambdas, cpu_cie_x_values, gpu_dynamic_pointers);
    cie_xyz[1] = Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
        cpu_cie_lambdas, cpu_cie_y_values, gpu_dynamic_pointers);
    cie_xyz[2] = Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
        cpu_cie_lambdas, cpu_cie_z_values, gpu_dynamic_pointers);

    auto cie_illum_d6500 = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(CIE_Illum_D6500), std::end(CIE_Illum_D6500)), true, cie_xyz[1],
        gpu_dynamic_pointers);

    GlobalSpectra *global_spectra;
    CHECK_CUDA_ERROR(cudaMallocManaged(&global_spectra, sizeof(GlobalSpectra)));
    gpu_dynamic_pointers.push_back(global_spectra);

    for (uint idx = 0; idx < 3; ++idx) {
        global_spectra->cie_xyz[idx] = cie_xyz[idx];
    }
    global_spectra->cie_y = cie_xyz[1];

    if (gamut == RGBtoSpectrumData::Gamut::sRGB) {
        RGBtoSpectrumData::RGBtoSpectrumTable *rgb_to_spectrum_table;
        CHECK_CUDA_ERROR(cudaMallocManaged(&rgb_to_spectrum_table,
                                           sizeof(RGBtoSpectrumData::RGBtoSpectrumTable)));
        rgb_to_spectrum_table->init("sRGB", thread_pool);

        RGBColorSpace *rgb_color_space;
        CHECK_CUDA_ERROR(cudaMallocManaged(&rgb_color_space, sizeof(RGBColorSpace)));

        rgb_color_space->init(Point2f(0.64, 0.33), Point2f(0.3, 0.6), Point2f(0.15, 0.06),
                              cie_illum_d6500, rgb_to_spectrum_table, cie_xyz);

        global_spectra->rgb_color_space = rgb_color_space;

        gpu_dynamic_pointers.push_back(rgb_to_spectrum_table);
        gpu_dynamic_pointers.push_back(rgb_color_space);

    } else {
        REPORT_FATAL_ERROR();
    }

    const std::chrono::duration<FloatType> duration{std::chrono::system_clock::now() - start};
    std::cout << std::fixed << std::setprecision(1) << "spectra computing took " << duration.count()
              << " seconds.\n"
              << std::flush;

    return global_spectra;
}
