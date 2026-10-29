#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <ext/lodepng/lodepng.h>
#include <ext/tinyexr/tinyexr.h>
#include <limits>
#include <new>
#include <pbrt/film/gbuffer.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectrum_util/color_encoding.h>
#include <stdexcept>
#include <vector>

GBuffer::GBuffer(Point2i resolution, GPUMemoryAllocator &allocator) : resolution(resolution) {
    if (resolution.x <= 0 || resolution.y <= 0) {
        throw std::runtime_error("GBuffer requires a positive resolution");
    }

    const size_t num_pixels = size_t(resolution.x) * resolution.y;
    pixels = allocator.allocate<GBufferPixel>(num_pixels);
    for (size_t i = 0; i < num_pixels; ++i) {
        new (pixels + i) GBufferPixel;
    }
}

void GBuffer::write_to_exr(const std::string &filename) const {
    constexpr int num_channels = 9;
    // EXR channels are stored in lexicographic order, with matching planar data.
    constexpr std::array<const char *, num_channels> names = {
        "albedo.B", "albedo.G", "albedo.R", "ao", "coverage", "depth", "normal.X", "normal.Y", "normal.Z",
    };

    const size_t num_pixels = size_t(resolution.x) * resolution.y;
    std::array<std::vector<float>, num_channels> channel_pixels;
    for (auto &values : channel_pixels) {
        values.resize(num_pixels);
    }
    for (size_t i = 0; i < num_pixels; ++i) {
        const auto &pixel = pixels[i];
        channel_pixels[0][i] = float(pixel.albedo.b);
        channel_pixels[1][i] = float(pixel.albedo.g);
        channel_pixels[2][i] = float(pixel.albedo.r);
        channel_pixels[3][i] = float(pixel.ambient_occlusion);
        channel_pixels[4][i] = float(pixel.coverage);
        channel_pixels[5][i] = float(pixel.depth);
        channel_pixels[6][i] = float(pixel.normal.x);
        channel_pixels[7][i] = float(pixel.normal.y);
        channel_pixels[8][i] = float(pixel.normal.z);
    }

    std::array<unsigned char *, num_channels> image_data;
    std::array<EXRChannelInfo, num_channels> channels{};
    std::array<int, num_channels> pixel_types;
    for (int c = 0; c < num_channels; ++c) {
        image_data[c] = reinterpret_cast<unsigned char *>(channel_pixels[c].data());
        std::strcpy(channels[c].name, names[c]);
        channels[c].pixel_type = TINYEXR_PIXELTYPE_FLOAT;
        channels[c].x_sampling = 1;
        channels[c].y_sampling = 1;
        pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;
    }

    EXRImage image;
    InitEXRImage(&image);
    image.num_channels = num_channels;
    image.images = image_data.data();
    image.width = resolution.x;
    image.height = resolution.y;

    EXRHeader header;
    InitEXRHeader(&header);
    header.num_channels = num_channels;
    header.channels = channels.data();
    header.pixel_types = pixel_types.data();
    header.requested_pixel_types = pixel_types.data();
    header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;

    const char *error = nullptr;
    if (SaveEXRImageToFile(&image, &header, filename.c_str(), &error) != TINYEXR_SUCCESS) {
        const std::string message = error ? error : "unknown error";
        if (error) {
            FreeEXRErrorMessage(error);
        }
        throw std::runtime_error("Cannot write G-buffer EXR '" + filename + "': " + message);
    }
}

namespace {

uint8_t encode_unit(double value) {
    if (!std::isfinite(value)) {
        return 0;
    }
    return uint8_t(std::lround(std::clamp(value, 0.0, 1.0) * 255));
}

bool has_surface(const GBufferPixel &pixel) {
    return std::isfinite(pixel.coverage) && pixel.coverage > 0;
}

bool has_depth(const GBufferPixel &pixel) {
    return has_surface(pixel) && std::isfinite(pixel.depth) && pixel.depth > 0;
}

} // namespace

void GBuffer::copy_to_frame_buffer(uint8_t *frame_buffer, GBufferChannel channel) const {
    if (!frame_buffer) {
        throw std::runtime_error("GBuffer frame buffer is null");
    }

    const size_t num_pixels = size_t(resolution.x) * resolution.y;
    double min_log_depth = std::numeric_limits<double>::infinity();
    double max_log_depth = -std::numeric_limits<double>::infinity();
    if (channel == GBufferChannel::Depth) {
        for (size_t i = 0; i < num_pixels; ++i) {
            if (has_depth(pixels[i])) {
                const double log_depth = std::log1p(double(pixels[i].depth));
                min_log_depth = std::min(min_log_depth, log_depth);
                max_log_depth = std::max(max_log_depth, log_depth);
            }
        }
    }

    const SRGBColorEncoding srgb_encoding;
    for (size_t i = 0; i < num_pixels; ++i) {
        auto *encoded = frame_buffer + 3 * i;
        encoded[0] = encoded[1] = encoded[2] = 0;
        const auto &pixel = pixels[i];
        if (!has_surface(pixel)) {
            continue;
        }

        switch (channel) {
        case GBufferChannel::AmbientOcclusion: {
            encoded[0] = encoded[1] = encoded[2] = encode_unit(pixel.ambient_occlusion);
            break;
        }
        case GBufferChannel::SurfaceNormal: {
            const double length = std::hypot(double(pixel.normal.x), double(pixel.normal.y), double(pixel.normal.z));
            if (std::isfinite(length) && length > 0) {
                encoded[0] = encode_unit(0.5 + 0.5 * double(pixel.normal.x) / length);
                encoded[1] = encode_unit(0.5 + 0.5 * double(pixel.normal.y) / length);
                encoded[2] = encode_unit(0.5 + 0.5 * double(pixel.normal.z) / length);
            }
            break;
        }
        case GBufferChannel::Albedo: {
            for (int c = 0; c < 3; ++c) {
                const auto value = pixel.albedo[c];
                encoded[c] = std::isfinite(value) ? srgb_encoding.from_linear(value) : 0;
            }
            break;
        }
        case GBufferChannel::Depth: {
            if (has_depth(pixel)) {
                double value = 1;
                if (max_log_depth > min_log_depth) {
                    value = 1 - (std::log1p(double(pixel.depth)) - min_log_depth) /
                                    (max_log_depth - min_log_depth);
                }
                encoded[0] = encoded[1] = encoded[2] = encode_unit(value);
            }
            break;
        }
        }
    }
}

void GBuffer::write_to_png(const std::string &filename, GBufferChannel channel) const {
    std::vector<unsigned char> encoded(size_t(resolution.x) * resolution.y * 3);
    copy_to_frame_buffer(encoded.data(), channel);
    const unsigned error = lodepng::encode(filename, encoded, resolution.x, resolution.y, LCT_RGB, 8);
    if (error) {
        throw std::runtime_error("Cannot write G-buffer PNG '" + filename + "': " + lodepng_error_text(error));
    }
}
