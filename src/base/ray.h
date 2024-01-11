//
// Created by wentao on 4/6/23.
//

#ifndef CUDA_RAY_TRACER_RAY_H
#define CUDA_RAY_TRACER_RAY_H

#include "base/point.h"
#include "base/vector3.h"

class Ray {
    public:
        Point o;
        Vector3 d;

        __device__ Ray() {}

        __device__ Ray(const Point _o, const Vector3 _d) : o(_o), d(_d) {}

        __device__ Point at(float t) const {
            return o + t * d;
        }
};

#endif // CUDA_RAY_TRACER_RAY_H
