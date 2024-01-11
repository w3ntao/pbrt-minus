#pragma once

#include "base/point.h"
#include "base/vector3.h"

class Ray {
    public:
        Point o;
        Vector3 d;

        PBRT_GPU Ray() {}

        PBRT_GPU Ray(const Point _o, const Vector3 _d) : o(_o), d(_d) {}

        PBRT_GPU Point at(double t) const {
            return o + t * d;
        }
};
