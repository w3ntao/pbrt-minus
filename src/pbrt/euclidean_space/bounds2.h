#pragma once

#include <pbrt/euclidean_space/point2.h>
#include <pbrt/euclidean_space/vector2.h>
#include <pbrt/util/math.h>
#include <vector>

// Bounds2 Definition
template <typename T>
class Bounds2 {
  public:
    // Bounds2 Public Methods
    PBRT_CPU_GPU
    Bounds2() {
        T minimum = std::numeric_limits<T>::lowest();
        T maximum = std::numeric_limits<T>::max();
        p_min = Point2<T>(maximum, maximum);
        p_max = Point2<T>(minimum, minimum);
    }

    PBRT_CPU_GPU
    explicit Bounds2(Point2<T> p) : p_min(p), p_max(p) {}

    PBRT_CPU_GPU
    Bounds2(Point2<T> p1, Point2<T> p2) : p_min(p1.min(p2)), p_max(p1.max(p2)) {}

    PBRT_CPU_GPU
    Vector2<T> Diagonal() const {
        return p_max - p_min;
    }

    PBRT_CPU_GPU
    Point2<T> lerp(Point2f t) const {
        return Point2<T>(pbrt::lerp(t.x, p_min.x, p_max.x), pbrt::lerp(t.y, p_min.y, p_max.y));
    }

    PBRT_CPU_GPU
    T area() const {
        return (p_max.x - p_min.x) * (p_max.y - p_min.y);
    }

    PBRT_CPU_GPU
    bool is_empty() const {
        return p_min.x >= p_max.x || p_min.y >= p_max.y;
    }

    PBRT_CPU_GPU
    bool is_degenerate() const {
        return p_min.x > p_max.x || p_min.y > p_max.y;
    }

    PBRT_CPU_GPU
    uint8_t max_dimension() const {
        Vector2<T> diag = Diagonal();
        return diag.x > diag.y ? 0 : 1;
    }

    PBRT_CPU_GPU
    Point2<T> operator[](uint8_t i) const {
        return (i == 0) ? p_min : p_max;
    }

    PBRT_CPU_GPU
    Point2<T> &operator[](uint8_t i) {
        return (i == 0) ? p_min : p_max;
    }

    PBRT_CPU_GPU
    bool operator==(const Bounds2<T> &b) const {
        return b.p_min == p_min && b.p_max == p_max;
    }

    PBRT_CPU_GPU
    bool operator!=(const Bounds2<T> &b) const {
        return b.p_min != p_min || b.p_max != p_max;
    }

    PBRT_CPU_GPU
    Point2<T> corner(int corner) const {
        return Point2<T>((*this)[(corner & 1)].x, (*this)[(corner & 2) ? 1 : 0].y);
    }

    PBRT_CPU_GPU
    Vector2<T> offset(Point2<T> p) const {
        Vector2<T> o = p - p_min;
        if (p_max.x > p_min.x) {
            o.x /= p_max.x - p_min.x;
        }
        if (p_max.y > p_min.y) {
            o.y /= p_max.y - p_min.y;
        }
        return o;
    }

    PBRT_CPU_GPU
    Bounds2 intersect(const Bounds2 &b) const {
        auto p_min = this->p_min.max(b.p_min);
        auto p_max = this->p_max.min(b.p_max);

        if (p_min.x > p_max.x || p_min.y > p_max.y) {
            return Bounds2(Point2<T>(NAN, NAN));
        }

        Bounds2 result;
        result.p_min = p_min;
        result.p_max = p_max;

        return result;
    }

    PBRT_CPU_GPU
    bool contain(const Point2<T> pt) const {
        return pt.x >= p_min.x && pt.x <= p_max.x && pt.y >= p_min.y && pt.y <= p_max.y;
    }

    [[nodiscard]] std::vector<Point2i> range() const {
        static_assert(std::is_same<T, int>::value);

        std::vector<Point2i> result;
        result.reserve(this->area());

        for (int y = p_min.y; y < p_max.y; ++y) {
            for (int x = p_min.x; x < p_max.x; ++x) {
                result.push_back(Point2i(x, y));
            }
        }

        return result;
    }

    friend std::ostream &operator<<(std::ostream &stream, const Bounds2 &b) {
        stream << "Bounds2: (p_min: " << b.p_min << ", p_max: " << b.p_max << ")";
        return stream;
    }

    Point2<T> p_min;
    Point2<T> p_max;
};

using Bounds2i = Bounds2<int>;
using Bounds2f = Bounds2<Real>;
