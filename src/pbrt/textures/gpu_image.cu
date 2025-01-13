#include <ext/lodepng/lodepng.h>
#include <pbrt/spectrum_util/color_encoding.h>
#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/textures/gpu_image.h>
#include <filesystem>

// clang-format off
#define TINYEXR_IMPLEMENTATION
#include <ext/tinyexr/tinyexr.h>

#include <pbrt/gpu/gpu_memory_allocator.h>
// clang-format on

PBRT_CPU_GPU
Point2i remap_pixel_coord(const Point2i p, const Point2i resolution, WrapMode2D wrap_mode) {
    if (wrap_mode[0] == WrapMode::OctahedralSphere || wrap_mode[1] == WrapMode::OctahedralSphere) {
        auto coord = p;

        if (coord[0] < 0) {
            coord[0] = -coord[0];                    // mirror across u = 0
            coord[1] = resolution[1] - 1 - coord[1]; // mirror across v = 0.5
        } else if (coord[0] >= resolution[0]) {
            coord[0] = 2 * resolution[0] - 1 - coord[0]; // mirror across u = 1
            coord[1] = resolution[1] - 1 - coord[1];     // mirror across v = 0.5
        }

        if (coord[1] < 0) {
            coord[0] = resolution[0] - 1 - coord[0]; // mirror across u = 0.5
            coord[1] = -coord[1];                    // mirror across v = 0;
        } else if (coord[1] >= resolution[1]) {
            coord[0] = resolution[0] - 1 - coord[0];     // mirror across u = 0.5
            coord[1] = 2 * resolution[1] - 1 - coord[1]; // mirror across v = 1
        }

        // Bleh: things don't go as expected for 1x1 images.
        if (resolution[0] == 1) {
            coord[0] = 0;
        }

        if (resolution[1] == 1) {
            coord[1] = 0;
        }

        return coord;
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

const GPUImage *GPUImage::create_from_file(const std::string &filename,
                                           GPUMemoryAllocator &allocator) {
    auto image = allocator.allocate<GPUImage>();

    auto path = std::filesystem::path(filename);
    if (path.extension() == ".png") {
        image->init_png(filename, allocator);
        return image;
    }

    if (path.extension() == ".exr") {
        image->init_exr(filename, allocator);
        return image;
    }

    REPORT_FATAL_ERROR();
    return nullptr;
}

void GPUImage::init_png(const std::string &filename, GPUMemoryAllocator &allocator) {
    std::vector<unsigned char> rgba_pixels;
    uint width;
    uint height;

    unsigned error = lodepng::decode(rgba_pixels, width, height, filename);
    if (error) {
        std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        REPORT_FATAL_ERROR();
    }
    // the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as
    // texture,

    resolution = Point2i(width, height);

    auto gpu_pixels = allocator.allocate<RGB>(width * height);

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

void GPUImage::init_exr(const std::string &filename, GPUMemoryAllocator &allocator) {
    float *out; // width * height * RGBA
    int width;
    int height;
    const char *err = nullptr; // or nullptr in C++11
    if (LoadEXR(&out, &width, &height, filename.c_str(), &err) != TINYEXR_SUCCESS) {
        if (err) {
            fprintf(stderr, "ERR : %s\n", err);
            FreeEXRErrorMessage(err); // release memory of error message.
        }
        exit(1);
    }
    resolution = Point2i(width, height);

    auto gpu_pixels = allocator.allocate<RGB>(width * height);

    for (uint idx = 0; idx < width * height; ++idx) {
        auto r = out[idx * 4 + 0];
        auto g = out[idx * 4 + 1];
        auto b = out[idx * 4 + 2];
        // auto alpha = out[idx * 4 + 3];

        gpu_pixels[idx] = RGB(r, g, b);
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
