#pragma once

#include "pbrt/euclidean_space/point3.h"
#include "pbrt/euclidean_space/point3fi.h"
#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/euclidean_space/normal3f.h"

class Ray {
  public:
    Point3f o;
    Vector3f d;

    PBRT_CPU_GPU Ray() {}

    PBRT_CPU_GPU Ray(Point3f _o, Vector3f _d) : o(_o), d(_d) {}

    PBRT_CPU_GPU Point3f at(FloatType t) const {
        return o + t * d;
    }

    // Ray Inline Functions
    PBRT_CPU_GPU static Point3f offset_ray_origin(const Point3fi &pi, const Normal3f &n,
                                                  const Vector3f &w) {
        // Find vector _offset_ to corner of error bounds and compute initial _po_
        FloatType d = n.abs().dot(pi.error());
        auto offset = d * n.to_vector3();
        if (n.dot(w) < 0.0) {
            offset = -offset;
        }

        auto po = pi.to_point3f() + offset;

        // Round offset point _po_ away from _p_
        for (int i = 0; i < 3; ++i) {
            if (offset[i] > 0.0) {
                po[i] = next_float_up(po[i]);
            } else if (offset[i] < 0.0) {
                po[i] = next_float_down(po[i]);
            }
        }

        return po;
    }
};

class DifferentialRay {
  public:
    Ray ray;

    bool hasDifferentials;

    Point3f rxOrigin;
    Point3f ryOrigin;
    Vector3f rxDirection;
    Vector3f ryDirection;

    PBRT_CPU_GPU DifferentialRay(const Point3f o, const Vector3f d)
        : ray(Ray(o, d)), hasDifferentials(false) {}
};
