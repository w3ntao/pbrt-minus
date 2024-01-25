#pragma once

#include "pbrt/euclidean_space/point3.h"
#include "pbrt/euclidean_space/vector3.h"

class Ray {
  public:
    Point3f o;
    Vector3f d;

    PBRT_CPU_GPU Ray() {}

    PBRT_CPU_GPU Ray(const Point3f _o, const Vector3f _d) : o(_o), d(_d) {}

    PBRT_CPU_GPU Point3f at(double t) const {
        return o + t * d;
    }
};
