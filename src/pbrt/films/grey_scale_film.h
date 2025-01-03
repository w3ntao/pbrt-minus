#pragma once

#include "pbrt/euclidean_space/bounds2.h"
#include "pbrt/euclidean_space/point2.h"
#include <vector>

class Filter;

class GreyScaleFilm {
  public:
    GreyScaleFilm(const Point2i &_resolution)
        : resolution(_resolution),
          pixel_bound(Bounds2i(Point2i(0, 0), Point2i(_resolution.x, _resolution.y))),
          pixels(std::vector<FloatType>(_resolution.x * _resolution.y, 0)) {}

    void add_sample(const Point2i &coord, FloatType val) {
        pixels[coord.y * resolution.x + coord.x] += val;
    }

    void add_splat(const Point2f &p_film, FloatType val, const Filter *filter);

    void write_to_png(const std::string &filename) const;

  private:
    Point2i resolution;
    std::vector<FloatType> pixels;
    Bounds2i pixel_bound;
};
