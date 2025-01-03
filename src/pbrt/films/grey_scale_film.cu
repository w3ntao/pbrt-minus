#include "ext/lodepng/lodepng.h"
#include "pbrt/base/filter.h"
#include "pbrt/films/grey_scale_film.h"
#include "pbrt/spectrum_util/color_encoding.h"
#include "pbrt/spectrum_util/rgb.h"
#include "pbrt/util/basic_math.h"
#include <algorithm>

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

const std::vector palette_viridis_7 = {RGB(68, 1, 84) / 255,    RGB(68, 57, 131) / 255,
                                       RGB(49, 104, 142) / 255, RGB(33, 145, 140) / 255,
                                       RGB(53, 183, 121) / 255, RGB(144, 215, 67) / 251,
                                       RGB(253, 231, 37) / 255};

static RGB convert_to_heatmap_rgb(double linear) {
    auto colors = &rainbow_6;

    /*
    const auto gamma = 2.0;
    linear = pow(linear, 1.0 / gamma);
    */

    const auto gap = 1.0 / (colors->size() - 1);

    for (int idx = 0; idx < colors->size() - 1; ++idx) {
        if (linear >= gap * idx && linear < gap * (idx + 1)) {
            auto x = 1 - (linear - gap * idx) / gap;
            return colors->at(idx) * x + colors->at(idx + 1) * (1 - x);
        }
    }

    return colors->at(colors->size() - 1);
}

void GreyScaleFilm::write_to_png(const std::string &filename) const {
    auto sorted_value = pixels;
    std::sort(sorted_value.begin(), sorted_value.end(), std::greater{});

    const auto one_percent = 0.01;
    const double top_one_percent_max_intensity = sorted_value[pixels.size() * one_percent];

    if (top_one_percent_max_intensity <= 0) {
        REPORT_FATAL_ERROR();
    }

    SRGBColorEncoding srgb_encoding;

    std::vector<unsigned char> png_pixels(pixels.size() * 4);

    for (int idx = 0; idx < pixels.size(); ++idx) {
        auto val = clamp<double>(pixels[idx] / top_one_percent_max_intensity, 0, 1);

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

void GreyScaleFilm::add_splat(const Point2f &p_film, FloatType val, const Filter *filter) {
    if (filter == nullptr) {
        REPORT_FATAL_ERROR();
    }

    Point2f pDiscrete = p_film + Vector2f(0.5, 0.5);

    Vector2f radius = filter->get_radius();

    Bounds2i splatBounds((pDiscrete - radius).floor(),
                         (pDiscrete + radius).floor() + Vector2i(1, 1));

    splatBounds = splatBounds.intersect(pixel_bound);

    for (const auto pi : splatBounds.range()) {
        auto weight =
            std::abs(filter->evaluate(Point2f(p_film - pi.to_point2f() - Vector2f(0.5, 0.5))));
        // note: with std::abs(), it contributes differently than the original Film::add_splat()

        if (weight == 0) {
            continue;
        }

        auto pixel_index = pi.x + pi.y * resolution.x;
        pixels[pixel_index] += weight * val;
    }
}
