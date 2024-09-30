#pragma once

#include "pbrt/euclidean_space/point2.h"
#include <vector>

class GreyScaleFilm {
  public:
    GreyScaleFilm(const Point2i &_resolution)
        : resolution(_resolution),
          pixels(std::vector<FloatType>(_resolution.x * _resolution.y, 0)) {}

    void add_sample(const Point2i &coord, FloatType val) {
        pixels[coord.x + resolution.x * coord.y] += val;
    }

    void write_to_png(const std::string &filename);

  private:
    Point2i resolution;
    std::vector<FloatType> pixels;
};
