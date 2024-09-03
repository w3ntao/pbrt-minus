#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/util/macro.h"

PBRT_GPU
Point2f EqualAreaSphereToSquare(Vector3f v);

PBRT_CPU_GPU
Vector3f EqualAreaSquareToSphere(Point2f p);
