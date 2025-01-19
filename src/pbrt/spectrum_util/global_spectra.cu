#include <chrono>
#include <pbrt/base/spectrum.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/spectrum_util/rgb_color_space.h>

const GlobalSpectra *GlobalSpectra::create(RGBtoSpectrumData::Gamut gamut,
                                           GPUMemoryAllocator &allocator) {
    const auto start = std::chrono::system_clock::now();

    const auto vec_cie_lambdas = std::vector(std::begin(CIE_LAMBDA_CPU), std::end(CIE_LAMBDA_CPU));

    const auto vec_cie_x_values =
        std::vector(std::begin(CIE_X_VALUE_CPU), std::end(CIE_X_VALUE_CPU));
    const auto vec_cie_y_values =
        std::vector(std::begin(CIE_Y_VALUE_CPU), std::end(CIE_Y_VALUE_CPU));
    const auto vec_cie_z_values =
        std::vector(std::begin(CIE_Z_VALUE_CPU), std::end(CIE_Z_VALUE_CPU));

    const Spectrum *cie_xyz[3] = {nullptr, nullptr, nullptr};
    cie_xyz[0] = Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
        vec_cie_lambdas, vec_cie_x_values, allocator);
    cie_xyz[1] = Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
        vec_cie_lambdas, vec_cie_y_values, allocator);
    cie_xyz[2] = Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
        vec_cie_lambdas, vec_cie_z_values, allocator);

    const auto spectrum_cie_illum_d6500 =
        Spectrum::create_piecewise_linear_spectrum_from_interleaved(CIE_Illum_D6500, true,
                                                                    cie_xyz[1], allocator);

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
                              spectrum_cie_illum_d6500, rgb_to_spectrum_table, cie_xyz);

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
