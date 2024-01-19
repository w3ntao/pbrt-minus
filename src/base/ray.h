#pragma once

#include "euclidean_space/point3.h"
#include "euclidean_space/vector3.h"

class Ray {
  public:
    Point3f o;
    Vector3f d;

    PBRT_GPU Ray() {}

    PBRT_GPU Ray(const Point3f _o, const Vector3f _d) : o(_o), d(_d) {}

    PBRT_GPU Point3f at(double t) const {
        return o + t * d;
    }
};
