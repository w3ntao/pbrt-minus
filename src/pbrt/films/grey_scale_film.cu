#include "ext/lodepng/lodepng.h"
#include "pbrt/films/grey_scale_film.h"
#include "pbrt/spectrum_util/color_encoding.h"
#include "pbrt/spectrum_util/rgb.h"
#include "pbrt/util/basic_math.h"

/*
const std::vector<RGB> colors = {
    RGB(0, 0, 1), RGB(0, 1, 1), RGB(0, 1, 0), RGB(1, 1, 0), RGB(1, 0, 0),
};
const std::vector<RGB> colors = {RGB(0, 0, 0), RGB(0, 0, 1), RGB(0, 1, 1), RGB(0, 1, 0),
                                 RGB(1, 1, 0), RGB(1, 0, 0), RGB(1, 1, 1)};

const std::vector<RGB> colors = {
    RGB(68, 1, 84) / 255,   RGB(59, 82, 139) / 255,  RGB(33, 145, 140) / 255,
    RGB(94, 201, 98) / 255, RGB(253, 231, 37) / 255,
};
*/

const std::vector<RGB> colors = {RGB(68, 1, 84) / 255,    RGB(68, 57, 131) / 255,
                                 RGB(49, 104, 142) / 255, RGB(33, 145, 140) / 255,
                                 RGB(53, 183, 121) / 255, RGB(144, 215, 67) / 255,
                                 RGB(253, 231, 37) / 255};

/*
taken from https://www.andrewnoske.com/wiki/Code_-_heatmaps_and_color_gradients
5 colors rainbow:        blue, cyan, green, yellow, red
7 colors rainbow: black, blue, cyan, green, yellow, red, white
7 color viridis: https://waldyrious.net/viridis-palette-generator/
*/

static RGB convert_to_heatmap_rgb(double linear) {
    // linear = linear * 2;

    linear = clamp<double>(linear, 0, 1);
    const auto gamma = 2.0;
    linear = pow(linear, 1.0 / gamma);

    const auto gap = 1.0 / (colors.size() - 1);

    for (int idx = 0; idx < colors.size() - 1; ++idx) {
        if (linear >= gap * idx && linear < gap * (idx + 1)) {
            auto x = 1 - (linear - gap * idx) / gap;
            return colors[idx] * x + colors[idx + 1] * (1 - x);
        }
    }

    return colors[colors.size() - 1];
}

void GreyScaleFilm::write_to_png(const std::string &filename) {
    FloatType max_intensity = 0.0;
    for (float pixel : pixels) {
        max_intensity = std::max(max_intensity, pixel);
    }

    SRGBColorEncoding srgb_encoding;

    std::vector<unsigned char> png_pixels(pixels.size() * 4);

    for (int idx = 0; idx < pixels.size(); ++idx) {
        auto val = pixels[idx] / max_intensity;

        auto rgb = convert_to_heatmap_rgb(val);

        png_pixels[4 * idx + 0] = srgb_encoding.from_linear(rgb.r);
        png_pixels[4 * idx + 1] = srgb_encoding.from_linear(rgb.g);
        png_pixels[4 * idx + 2] = srgb_encoding.from_linear(rgb.b);
        png_pixels[4 * idx + 3] = 255;
    }

    // Encode the image
    // if there's an error, display it
    if (unsigned error = lodepng::encode(filename, png_pixels, resolution.x, resolution.y); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }
}
