//
// Created by wentao on 4/6/23.
//

#ifndef CUDA_RAY_TRACER_IMAGE_H
#define CUDA_RAY_TRACER_IMAGE_H

#include <vector>
#include <string>
#include "constants.h"
#include "base/color.h"

class Image {
    private:
        std::vector<Color> pixels;
        uint width, height;

    public:
        Image() : width(0), height(0) {}

        Image(uint _width, uint _height) : width(_width), height(_height) {
            pixels = std::vector<Color>(_width * _height);
        }

        Image(const Color *frame_buffer, uint _width, uint _height) : width(_width), height(_height) {
            pixels = std::vector<Color>(_width * _height);
            for (uint x = 0; x < width; ++x) {
                for (uint y = 0; y < height; ++y) {
                    size_t pixel_index = y * width + x;
                    (*this)(x, y) = frame_buffer[pixel_index];
                }
            }
        }

        Image &operator=(const Image &other) {
            width = other.width;
            height = other.height;
            pixels = other.pixels;
            return *this;
        }

        void flip() {
            std::vector<Color> flipped_pixels = std::vector<Color>(width * height);

            for (uint x = 0; x < width; ++x) {
                for (uint y = 0; y < height; ++y) {
                    uint pixel_index = y * width + x;
                    uint flipped_index = (height - 1 - y) * width + x;

                    flipped_pixels[flipped_index] = pixels[pixel_index];
                }
            }
            pixels = flipped_pixels;
        }

        void create(uint _width, uint _height) {
            width = _width;
            height = _height;
            pixels = std::vector<Color>(_width * _height);
        }

        Color &operator()(uint x, uint y) {
            return pixels[y * width + x];
        }

        const Color &operator()(uint x, uint y) const {
            return pixels[y * width + x];
        }

        void writePNG(const std::string &file_name);

        void readPNG(const std::string &file_name);
};
#endif // CUDA_RAY_TRACER_IMAGE_H
