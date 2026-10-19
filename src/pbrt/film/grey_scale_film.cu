#include <algorithm>
#include <cmath>
#include <ext/lodepng/lodepng.h>
#include <pbrt/base/filter.h>
#include <pbrt/film/grey_scale_film.h>
#include <pbrt/spectrum_util/color_encoding.h>
#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/util/math.h>

const auto black = RGB(0, 0, 0);
const auto blue = RGB(0, 0, 1);
const auto cyan = RGB(0, 1, 1);
const auto green = RGB(0, 1, 0);
const auto yellow = RGB(1, 1, 0);
const auto red = RGB(1, 0, 0);
const auto white = RGB(1, 1, 0);

/*
taken from https://www.andrewnoske.com/wiki/Code_-_heatmaps_and_color_gradients
5 colors rainbow:        blue, cyan, green, yellow, red
6 colors rainbow: black, blue, cyan, green, yellow, red
7 colors rainbow: black, blue, cyan, green, yellow, red, white
7 color viridis: https://waldyrious.net/viridis-palette-generator/
*/

const std::vector rainbow_5 = {
    blue, cyan, green, yellow, red,
};

const std::vector rainbow_6 = {
    black, blue, cyan, green, yellow, red,
};

const std::vector rainbow_7 = {black, blue, cyan, green, yellow, red, white};

const std::vector palette_viridis_7 = {RGB(68, 1, 84) / 255,    RGB(68, 57, 131) / 255,  RGB(49, 104, 142) / 255,
                                       RGB(33, 145, 140) / 255, RGB(53, 183, 121) / 255, RGB(144, 215, 67) / 251,
                                       RGB(253, 231, 37) / 255};

static RGB convert_to_heatmap_rgb(double linear) {
    // auto colors = &rainbow_6;
    auto colors = &palette_viridis_7;

    const auto gap = 1.0 / (colors->size() - 1);

    for (int idx = 0; idx < colors->size() - 1; ++idx) {
        if (linear >= gap * idx && linear < gap * (idx + 1)) {
            auto x = 1 - (linear - gap * idx) / gap;
            return colors->at(idx) * x + colors->at(idx + 1) * (1 - x);
        }
    }

    return colors->at(colors->size() - 1);
}

void GreyScaleFilm::write_to_png(const std::string &filename, bool grey) const {
    if (grey) {
        Real maximum = 0;
        for (const auto value : pixels) {
            if (std::isfinite(value) && value > maximum) {
                maximum = value;
            }
        }

        GreyScaleFilm normalized(resolution);
        if (maximum > 0) {
            for (size_t idx = 0; idx < pixels.size(); ++idx) {
                const auto value = pixels[idx];
                if (std::isfinite(value) && value > 0) {
                    normalized.pixels[idx] = value / maximum;
                }
            }
        }
        normalized.write_mask_to_png(filename);
        return;
    }

    auto sorted_value = pixels;
    std::sort(sorted_value.begin(), sorted_value.end(), std::greater{});

    const auto one_percent = 0.01;

    double max_intensity_threshold = 1;
    for (int i = pixels.size() * one_percent; i >= 0; i--) {
        if (sorted_value[i] > 0) {
            max_intensity_threshold = sorted_value[i];
            break;
        }
    }

    std::vector<unsigned char> png_pixels(pixels.size() * 4);

    SRGBColorEncoding srgb_encoding;
    for (int idx = 0; idx < pixels.size(); ++idx) {
        auto val = clamp<double>(pixels[idx] / max_intensity_threshold, 0, 1);

        auto rgb = convert_to_heatmap_rgb(val);

        png_pixels[4 * idx + 0] = srgb_encoding.from_linear(rgb.r);
        png_pixels[4 * idx + 1] = srgb_encoding.from_linear(rgb.g);
        png_pixels[4 * idx + 2] = srgb_encoding.from_linear(rgb.b);
        png_pixels[4 * idx + 3] = 255;
    }

    // Encode the image
    // if there's an error, display it
    if (unsigned error = lodepng::encode(filename, png_pixels, resolution.x, resolution.y); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }
}

void GreyScaleFilm::write_mask_to_png(const std::string &filename) const {
    std::vector<unsigned char> png_pixels(pixels.size());
    for (size_t idx = 0; idx < pixels.size(); ++idx) {
        const auto value = std::isnan(pixels[idx]) ? 0.0 : clamp<double>(pixels[idx], 0, 1);
        png_pixels[idx] = static_cast<unsigned char>(std::lround(value * 255));
    }

    if (unsigned error = lodepng::encode(filename, png_pixels, resolution.x, resolution.y, LCT_GREY, 8); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }
}
