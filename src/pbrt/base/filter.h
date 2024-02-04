#pragma once

struct FilterSample {
    Point2f p;
    double weight;

    PBRT_GPU FilterSample(const Point2f &_p, double _weight) : p(_p), weight(_weight) {}
};

class Filter {
  public:
    PBRT_GPU
    virtual FilterSample sample(Point2f u) const = 0;
};
