#pragma once

#include <pbrt/euclidean_space/interval.h>
#include <pbrt/euclidean_space/point3.h>

class Point3fi : public Point3<Interval> {
  public:
    using Point3<Interval>::x;
    using Point3<Interval>::y;
    using Point3<Interval>::z;
    using Point3<Interval>::operator+;
    using Point3<Interval>::operator*;
    using Point3<Interval>::operator*=;

    PBRT_CPU_GPU
    Point3fi() : Point3<Interval>(Interval(NAN), Interval(NAN), Interval(NAN)) {}

    PBRT_CPU_GPU
    Point3fi(Interval x, Interval y, Interval z) : Point3<Interval>(x, y, z) {}

    PBRT_CPU_GPU
    Point3fi(Real x, Real y, Real z)
        : Point3<Interval>(Interval(x), Interval(y), Interval(z)) {}

    PBRT_CPU_GPU
    Point3fi(const Point3f &p) : Point3<Interval>(Interval(p.x), Interval(p.y), Interval(p.z)) {}

    PBRT_CPU_GPU
    Point3fi(Point3<Interval> p) : Point3<Interval>(p) {}

    PBRT_CPU_GPU
    Point3fi(Point3f p, Vector3f e)
        : Point3<Interval>(Interval::from_value_and_error(p.x, e.x),
                           Interval::from_value_and_error(p.y, e.y),
                           Interval::from_value_and_error(p.z, e.z)) {}

    PBRT_CPU_GPU Point3f to_point3f() const {
        return Point3f(x.midpoint(), y.midpoint(), z.midpoint());
    }

    PBRT_CPU_GPU
    bool has_nan() const {
        return x.has_nan() || y.has_nan() || z.has_nan();
    }

    PBRT_CPU_GPU Vector3f error() const {
        return {x.width() / 2, y.width() / 2, z.width() / 2};
    }

    PBRT_CPU_GPU
    bool is_exact() const {
        return x.width() == 0 && y.width() == 0 && z.width() == 0;
    }

    template <typename T>
    PBRT_CPU_GPU void operator+=(Vector3<T> v) {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    template <typename U>
    PBRT_CPU_GPU Point3fi operator-(Vector3<U> v) const {
        return {x - v.x, y - v.y, z - v.z};
    }

    PBRT_CPU_GPU Point3fi operator/(Real divisor) const {
        return {x / divisor, y / divisor, z / divisor};
    }
};
