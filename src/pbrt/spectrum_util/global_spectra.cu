#include <pbrt/base/spectrum.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/spectrum_util/rgb_color_space.h>
#include <chrono>

const GlobalSpectra *GlobalSpectra::create(RGBtoSpectrumData::Gamut gamut,
                                           GPUMemoryAllocator &allocator) {
    const auto start = std::chrono::system_clock::now();

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
        cpu_cie_lambdas, cpu_cie_x_values, allocator);
    cie_xyz[1] = Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
        cpu_cie_lambdas, cpu_cie_y_values, allocator);
    cie_xyz[2] = Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
        cpu_cie_lambdas, cpu_cie_z_values, allocator);

    auto cie_illum_d6500 = Spectrum::create_piecewise_linear_spectrum_from_interleaved(
        std::vector(std::begin(CIE_Illum_D6500), std::end(CIE_Illum_D6500)), true, cie_xyz[1],
        allocator);

    auto global_spectra = allocator.allocate<GlobalSpectra>();

    for (uint idx = 0; idx < 3; ++idx) {
        global_spectra->cie_xyz[idx] = cie_xyz[idx];
    }
    global_spectra->cie_y = cie_xyz[1];

    if (gamut == RGBtoSpectrumData::Gamut::sRGB) {
        auto rgb_to_spectrum_table = allocator.allocate<RGBtoSpectrumData::RGBtoSpectrumTable>();
        rgb_to_spectrum_table->init("sRGB");

        auto rgb_color_space = allocator.allocate<RGBColorSpace>();
        rgb_color_space->init(Point2f(0.64, 0.33), Point2f(0.3, 0.6), Point2f(0.15, 0.06),
                              cie_illum_d6500, rgb_to_spectrum_table, cie_xyz);

        global_spectra->rgb_color_space = rgb_color_space;

    } else {
        REPORT_FATAL_ERROR();
    }

    const std::chrono::duration<FloatType> duration{std::chrono::system_clock::now() - start};
    std::cout << std::fixed << std::setprecision(1) << "spectra computing took " << duration.count()
              << " seconds.\n"
              << std::flush;

    return global_spectra;
}
