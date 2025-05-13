#pragma once

#include <pbrt/euclidean_space/normal3f.h>
#include <pbrt/euclidean_space/point3.h>
#include <pbrt/euclidean_space/point3fi.h>
#include <pbrt/euclidean_space/vector3.h>

class Ray {
  public:
    Point3f o;
    Vector3f d;

    PBRT_CPU_GPU Ray() {}

    PBRT_CPU_GPU Ray(Point3f _o, Vector3f _d) : o(_o), d(_d) {}

    PBRT_CPU_GPU Point3f at(Real t) const {
        return o + t * d;
    }

    // Ray Inline Functions
    PBRT_CPU_GPU
    static Point3f offset_ray_origin(const Point3fi &pi, const Normal3f &n, const Vector3f &w) {
        // Find vector _offset_ to corner of error bounds and compute initial _po_
        Real d = n.abs_dot(pi.error());
        auto offset = d * n.to_vector3();
        if (n.dot(w) < 0) {
            offset = -offset;
        }

        auto po = pi.to_point3f() + offset;

        // Round offset point _po_ away from _p_
        for (int i = 0; i < 3; ++i) {
            if (offset[i] > 0) {
                po[i] = next_float_up(po[i]);
            } else if (offset[i] < 0) {
                po[i] = next_float_down(po[i]);
            }
        }

        return po;
    }

    PBRT_CPU_GPU static Ray spawn_ray_to(const Point3fi &p_from, const Normal3f &n_from,
                                         const Point3fi &p_to, const Normal3f &n_to) {
        auto pf = offset_ray_origin(p_from, n_from, p_to.to_point3f() - p_from.to_point3f());
        auto pt = offset_ray_origin(p_to, n_to, pf - p_to.to_point3f());

        return Ray(pf, pt - pf);
    }
};
