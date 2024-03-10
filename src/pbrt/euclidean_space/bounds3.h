#pragma once

#include "pbrt/util/math.h"

template <typename T>
class Bounds3 {
  public:
    PBRT_CPU_GPU
    Bounds3() {
        T minimum = std::numeric_limits<T>::lowest();
        T maximum = std::numeric_limits<T>::max();
        p_min = Point3<T>(maximum, maximum, maximum);
        p_max = Point3<T>(minimum, minimum, minimum);
    }

    PBRT_CPU_GPU
    static Bounds3 empty() {
        return Bounds3();
    }

    PBRT_CPU_GPU
    Bounds3(const Bounds3 &b) : p_min(b.p_min), p_max(b.p_max) {}

    PBRT_CPU_GPU
    explicit Bounds3(Point3<T> p) : p_min(p), p_max(p) {}

    PBRT_CPU_GPU
    Bounds3(Point3<T> p1, Point3<T> p2) : p_min(p1.min(p2)), p_max(p1.max(p2)) {}

    PBRT_CPU_GPU
    Bounds3(const Point3f *points, int n) {
        if (n == 0) {
            *this = Bounds3();
            return;
        }

        p_min = points[0];
        p_max = points[0];

        for (int i = 1; i < n; ++i) {
            p_min = p_min.min(points[i]);
            p_max = p_max.max(points[i]);
        }
    }

    PBRT_CPU_GPU
    bool is_empty() const {
        return p_min.x >= p_max.x || p_min.y >= p_max.y || p_min.z >= p_max.z;
    }

    PBRT_CPU_GPU
    bool operator==(const Bounds3 &bounds) const {
        return p_min == bounds.p_min && p_max == bounds.p_max;
    }

    PBRT_CPU_GPU
    Point3<T> operator[](uint index) const {
        switch (index) {
        case (0): {
            return p_min;
        }
        case (1): {
            return p_max;
        }
        default: {
            printf("Bounds3: invalid index `%d`\n\n", index);

#if defined(__CUDA_ARCH__)
            asm("trap;");
#else
            throw std::runtime_error("Bounds3: invalid index\n\n");
#endif
        }
        }
    }

    PBRT_CPU_GPU Bounds3 operator+(const Bounds3 &b) const {
        return Bounds3(p_min.min(b.p_min), p_max.max(b.p_max));
    }

    PBRT_CPU_GPU void operator+=(const Bounds3 &b) {
        *this = *this + b;
    }

    PBRT_CPU_GPU void operator+=(const Point3f &p) {
        p_min = p_min.min(p);
        p_max = p_max.max(p);
    }

    PBRT_CPU_GPU Vector3<T> diagonal() const {
        return p_max - p_min;
    }

    PBRT_CPU_GPU Point3<T> centroid() const {
        return 0.5 * (p_min + p_max);
    }

    PBRT_CPU_GPU
    Vector3f offset(Point3f p) const {
        Vector3f o = p - p_min;
        if (p_max.x > p_min.x) {
            o.x /= p_max.x - p_min.x;
        }

        if (p_max.y > p_min.y) {
            o.y /= p_max.y - p_min.y;
        }

        if (p_max.z > p_min.z) {
            o.z /= p_max.z - p_min.z;
        }

        return o;
    }

    PBRT_CPU_GPU double surface_area() const {
        auto d = diagonal();
        return 2.0 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    PBRT_CPU_GPU int8_t max_dimension() const {
        auto d = diagonal();
        if (d.x > d.y && d.x > d.z) {
            return 0;
        }
        if (d.y > d.z) {
            return 1;
        }
        return 2;
    }

    PBRT_GPU bool fast_intersect(const Ray &ray, double ray_t_max, const Vector3f &inv_dir,
                                 const int dir_is_neg[3]) const {
        // Check for ray intersection against $x$ and $y$ slabs

        auto o = ray.o;
        auto t_min = ((*this)[dir_is_neg[0]].x - o.x) * inv_dir.x;
        auto t_max = ((*this)[1 - dir_is_neg[0]].x - o.x) * inv_dir.x;

        auto ty_min = ((*this)[dir_is_neg[1]].y - o.y) * inv_dir.y;
        auto ty_max = ((*this)[1 - dir_is_neg[1]].y - o.y) * inv_dir.y;

        // Update _tMax_ and _tyMax_ to ensure robust bounds intersection
        t_max *= 1.0 + 2.0 * gamma(3);
        ty_max *= 1.0 + 2.0 * gamma(3);

        if (t_min > ty_max || ty_min > t_max) {
            return false;
        }

        if (ty_min > t_min) {
            t_min = ty_min;
        }

        if (ty_max < t_max) {
            t_max = ty_max;
        }

        // Check for ray intersection against $z$ slab

        auto tz_min = ((*this)[dir_is_neg[2]].z - o.z) * inv_dir.z;
        auto tz_max = ((*this)[1 - dir_is_neg[2]].z - o.z) * inv_dir.z;

        // Update _tzMax_ to ensure robust bounds intersection
        tz_max *= 1.0 + 2.0 * gamma(3);

        if (t_min > tz_max || tz_min > t_max) {
            return false;
        }

        if (tz_min > t_min) {
            t_min = tz_min;
        }

        if (tz_max < t_max) {
            t_max = tz_max;
        }

        return t_min < ray_t_max && t_max > 0.0;
    }

    Point3<T> p_min, p_max;
};

using Bounds3f = Bounds3<double>;
