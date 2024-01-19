#pragma once

#include "euclidean_space/point3.h"

// Point3fi Definition
class Point3fi : public Point3<Interval> {
  public:
    using Point3<Interval>::x;
    using Point3<Interval>::y;
    using Point3<Interval>::z;
    using Point3<Interval>::operator+;
    using Point3<Interval>::operator*;
    using Point3<Interval>::operator*=;

    Point3fi() = default;
    PBRT_CPU_GPU
    Point3fi(Interval x, Interval y, Interval z) : Point3<Interval>(x, y, z) {}
    PBRT_CPU_GPU
    Point3fi(double x, double y, double z)
        : Point3<Interval>(Interval(x), Interval(y), Interval(z)) {}
    PBRT_CPU_GPU
    Point3fi(const Point3f &p) : Point3<Interval>(Interval(p.x), Interval(p.y), Interval(p.z)) {}
    PBRT_CPU_GPU
    Point3fi(Point3<Interval> p) : Point3<Interval>(p) {}
    PBRT_CPU_GPU
    Point3fi(Point3f p, Vector3f e)
        : Point3<Interval>(Interval::FromValueAndError(p.x, e.x),
                           Interval::FromValueAndError(p.y, e.y),
                           Interval::FromValueAndError(p.z, e.z)) {}
};
