#include <pbrt/film/pixel_sensor.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectrum_util/color_encoding.h>

const PixelSensor *PixelSensor::create_cie_1931(const Spectrum *const cie_xyz[3],
                                                const RGBColorSpace *output_color_space,
                                                const Spectrum *sensor_illum, Real imaging_ratio,
                                                GPUMemoryAllocator &allocator) {
    auto xyz_from_sensor_rgb = SquareMatrix<3>::identity();
    if (sensor_illum) {
        auto source_white = sensor_illum->to_xyz(cie_xyz).xy();
        auto target_white = output_color_space->w;

        xyz_from_sensor_rgb = white_balance(source_white, target_white);
    }

    return allocator.create<PixelSensor>(cie_xyz[0], cie_xyz[1], cie_xyz[2], imaging_ratio,
                                         xyz_from_sensor_rgb);
}
