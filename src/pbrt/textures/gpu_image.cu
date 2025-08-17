#include <ext/lodepng/lodepng.h>
#include <filesystem>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectrum_util/color_encoding.h>
#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/textures/gpu_image.h>

// clang-format off
#define STB_IMAGE_IMPLEMENTATION
#include <ext/stb/stb_image.h>
#undef STB_IMAGE_IMPLEMENTATION

#define TINYEXR_IMPLEMENTATION
#include <fstream>
#include <ext/tinyexr/tinyexr.h>
#undef TINYEXR_IMPLEMENTATION
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

    for (int c = 0; c < 2; c++) {
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
    auto image = allocator.create<GPUImage>();

    const auto file_extension = std::filesystem::path(filename).extension();

    if (file_extension == ".exr") {
        image->init_exr(filename, allocator);

    } else if (file_extension == ".pfm") {
        image->init_pfm(filename, allocator);

    } else if (file_extension == ".png") {
        image->init_png(filename, allocator);

    } else if (file_extension == ".tga") {
        image->init_tga(filename, allocator);

    } else {
        printf("\nERROR: image extension `%s` not implemented\n", file_extension.c_str());
        REPORT_FATAL_ERROR();
    }

    if (image->pixels == nullptr || image->resolution == Point2i(0, 0)) {
        REPORT_FATAL_ERROR();
    }

    return image;
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

    auto gpu_pixels = allocator.allocate<RGB>(width * height);

    for (int idx = 0; idx < width * height; ++idx) {
        auto r = out[idx * 4 + 0];
        auto g = out[idx * 4 + 1];
        auto b = out[idx * 4 + 2];
        // auto alpha = out[idx * 4 + 3];

        gpu_pixels[idx] = RGB(r, g, b);
    }

    resolution = Point2i(width, height);
    pixels = gpu_pixels;
}

void GPUImage::init_pfm(const std::string &filename, GPUMemoryAllocator &allocator) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "ERROR: couldn't opening file: `" << filename << "`\n";
        REPORT_FATAL_ERROR();
    }

    std::string header;
    file >> header;

    if (header != "PF" && header != "Pf") {
        std::cerr << "ERROR: not a valid PFM file\n";
        REPORT_FATAL_ERROR();
    }

    int width = 0;
    int height = 0;

    // Read width and height
    file >> width >> height;

    // Read scale factor
    float scale;
    file >> scale;

    // Skip the newline after the scale
    file.ignore(1);

    // Check if the image is stored in little-endian format
    bool isLittleEndian = scale < 0;
    if (isLittleEndian) {
        scale = -scale; // Make scale positive
    }

    // Read the pixel data
    size_t num_pixels = width * height * 3;

    // Resize the image vector to hold the pixel data
    std::vector<Real> image(num_pixels);

    file.read(reinterpret_cast<char *>(image.data()), num_pixels * sizeof(float));

    // If the image is in big-endian format, we need to swap the byte order
    if (!isLittleEndian) {
        for (size_t i = 0; i < num_pixels; ++i) {
            uint32_t *pixel = reinterpret_cast<uint32_t *>(&image[i]);
            *pixel = (*pixel >> 24) | ((*pixel & 0x00FF0000) >> 8) | ((*pixel & 0x0000FF00) << 8) |
                     (*pixel << 24);
        }
    }

    auto gpu_pixels = allocator.allocate<RGB>(width * height);
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            const auto pfm_idx = (width - 1 - y) * width + x;
            const auto image_idx = y * width + x;

            auto r = image[pfm_idx * 3 + 0];
            auto g = image[pfm_idx * 3 + 1];
            auto b = image[pfm_idx * 3 + 2];

            gpu_pixels[image_idx] = RGB(r, g, b);
        }
    }

    resolution = Point2i(width, height);
    pixels = gpu_pixels;
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

    auto gpu_pixels = allocator.allocate<RGB>(width * height);

    SRGBColorEncoding encoding;
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            int idx = y * width + x;

            const auto r = rgba_pixels[idx * 4 + 0];
            const auto g = rgba_pixels[idx * 4 + 1];
            const auto b = rgba_pixels[idx * 4 + 2];

            gpu_pixels[idx] =
                RGB(encoding.to_linear(r), encoding.to_linear(g), encoding.to_linear(b));
        }
    }

    resolution = Point2i(width, height);
    pixels = gpu_pixels;
}

void GPUImage::init_tga(const std::string &filename, GPUMemoryAllocator &allocator) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char *img = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (img == NULL) {
        printf("\ncouldn't read image `%s`\n", filename.c_str());
        REPORT_FATAL_ERROR();
    }

    if (width * height * channels == 0) {
        REPORT_FATAL_ERROR();
    }
    if (channels != 3 && channels != 4) {
        REPORT_FATAL_ERROR();
    }

    SRGBColorEncoding encoding;
    auto gpu_pixels = allocator.allocate<RGB>(width * height);
    for (int idx = 0; idx < width * height; ++idx) {
        const auto r = img[idx * channels + 0];
        const auto g = img[idx * channels + 1];
        const auto b = img[idx * channels + 2];

        gpu_pixels[idx] = RGB(encoding.to_linear(r), encoding.to_linear(g), encoding.to_linear(b));
    }

    stbi_image_free(img);

    resolution = Point2i(width, height);
    pixels = gpu_pixels;
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
