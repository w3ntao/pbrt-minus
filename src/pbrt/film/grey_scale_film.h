#pragma once

#include <pbrt/euclidean_space/bounds2.h>
#include <pbrt/euclidean_space/point2.h>
#include <vector>

class GreyScaleFilm {
  public:
    GreyScaleFilm(const Point2i &_resolution)
        : resolution(_resolution),
          pixel_bound(Bounds2i(Point2i(0, 0), Point2i(_resolution.x, _resolution.y))),
          pixels(std::vector<Real>(_resolution.x * _resolution.y, 0)) {}

    void add_sample(const Point2i &coord, Real val) {
        pixels[coord.y * resolution.x + coord.x] += val;
    }

    void write_to_png(const std::string &filename) const;

  private:
    Point2i resolution;
    std::vector<Real> pixels;
    Bounds2i pixel_bound;
};
