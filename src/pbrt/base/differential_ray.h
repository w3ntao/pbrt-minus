#pragma once

#include "pbrt/base/ray.h"

class DifferentialRay : public Ray {
  public:
    // RayDifferential Public Members
    bool hasDifferentials = false;
    Point3f rxOrigin;
    Point3f ryOrigin;
    Vector3f rxDirection;
    Vector3f ryDirection;

    PBRT_GPU DifferentialRay(Point3f o, Vector3f d) : Ray(o, d) {}
};
