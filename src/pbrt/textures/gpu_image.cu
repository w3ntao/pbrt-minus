#include "pbrt/textures/gpu_image.h"

#include "pbrt/spectrum_util/color_encoding.h"
#include "pbrt/spectrum_util/rgb.h"

#include "ext/lodepng/lodepng.h"

PBRT_CPU_GPU
Point2i remap_pixel_coord(const Point2i p, const Point2i resolution, WrapMode2D wrap_mode) {
    if (wrap_mode[0] == WrapMode::OctahedralSphere || wrap_mode[1] == WrapMode::OctahedralSphere) {
        REPORT_FATAL_ERROR();
    }

    auto coord = p;

    for (uint c = 0; c < 2; c++) {
        if (coord[c] >= 0 && coord[c] < resolution[c]) {
            continue;
        }
        switch (wrap_mode[c]) {
        case (WrapMode::Repeat): {
            coord[c] = mod(coord[c], resolution[c]);
            break;
        }
        default: {
            REPORT_FATAL_ERROR();
        }
        }
    }

    return coord;
}

void GPUImage::init(const std::string &filename, std::vector<void *> &gpu_dynamic_pointers) {
    std::vector<unsigned char> rgba_pixels;
    unsigned width, height;

    unsigned error = lodepng::decode(rgba_pixels, width, height, filename);
    if (error) {
        std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        REPORT_FATAL_ERROR();
    }
    // the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as
    // texture,

    resolution = Point2i(width, height);

    RGB *gpu_pixels;
    CHECK_CUDA_ERROR(cudaMallocManaged(&gpu_pixels, sizeof(RGB) * width * height));
    gpu_dynamic_pointers.push_back(gpu_pixels);

    SRGBColorEncoding encoding;
    for (uint x = 0; x < width; ++x) {
        for (uint y = 0; y < height; ++y) {
            uint index = y * width + x;

            // clang-format off
            auto red   = rgba_pixels[index * 4 + 0];
            auto green = rgba_pixels[index * 4 + 1];
            auto blue  = rgba_pixels[index * 4 + 2];
            // clang-format on

            gpu_pixels[index] =
                RGB(encoding.to_linear(red), encoding.to_linear(green), encoding.to_linear(blue));
        }
    }

    pixels = gpu_pixels;
    pixel_format = PixelFormat::U256;
}

PBRT_CPU_GPU
RGB GPUImage::fetch_pixel(const Point2i _p, WrapMode2D wrap_mode) const {
    auto p = remap_pixel_coord(_p, resolution, wrap_mode);
    return pixels[p.y * resolution.x + p.x];
}

PBRT_CPU_GPU
RGB GPUImage::bilerp(const Point2f p, const WrapMode2D wrap) const {
    // Compute discrete pixel coordinates and offsets for _p_

    const auto x = p.x * resolution.x - 0.5;
    const auto y = p.y * resolution.y - 0.5;

    const auto xi = floor(x);
    const auto yi = floor(y);

    const auto dx = x - xi;
    const auto dy = y - yi;
    
    // Load pixel channel values and return bilinearly interpolated value
    RGB v[4] = {
        fetch_pixel(Point2i(xi, yi), wrap),
        fetch_pixel(Point2i(xi + 1, yi), wrap),
        fetch_pixel(Point2i(xi, yi + 1), wrap),
        fetch_pixel(Point2i(xi + 1, yi + 1), wrap),
    };

    return (1.0 - dx) * (1.0 - dy) * v[0] + dx * (1.0 - dy) * v[1] + (1.0 - dx) * dy * v[2] +
           dx * dy * v[3];
}
