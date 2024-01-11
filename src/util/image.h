#pragma once

#include <vector>
#include <string>
#include "common.h"
#include "base/color.h"

class Image {
    private:
        std::vector<Color> pixels;
        int width, height;

    public:
        Image() : width(0), height(0) {}

        Image(int _width, int _height) : width(_width), height(_height) {
            pixels = std::vector<Color>(_width * _height);
        }

        Image(const Color *frame_buffer, int _width, int _height) : width(_width), height(_height) {
            pixels = std::vector<Color>(_width * _height);
            for (int x = 0; x < width; ++x) {
                for (int y = 0; y < height; ++y) {
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

            for (int x = 0; x < width; ++x) {
                for (int y = 0; y < height; ++y) {
                    int pixel_index = y * width + x;
                    int flipped_index = (height - 1 - y) * width + x;

                    flipped_pixels[flipped_index] = pixels[pixel_index];
                }
            }
            pixels = flipped_pixels;
        }

        void create(int _width, int _height) {
            width = _width;
            height = _height;
            pixels = std::vector<Color>(_width * _height);
        }

        Color &operator()(int x, int y) {
            return pixels[y * width + x];
        }

        const Color &operator()(int x, int y) const {
            return pixels[y * width + x];
        }

        void writePNG(const std::string &file_name);

        void readPNG(const std::string &file_name);
};
