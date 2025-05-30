#pragma once

#include <pbrt/euclidean_space/vector3.h>

class Frame {
  public:
    Vector3f x;
    Vector3f y;
    Vector3f z;

    PBRT_CPU_GPU
    Frame() {
        for (int idx = 0; idx < 3; idx++) {
            x[idx] = NAN;
            y[idx] = NAN;
            z[idx] = NAN;
        }
    }

    PBRT_CPU_GPU
    Frame(Vector3f _x, Vector3f _y, Vector3f _z) : x(_x), y(_y), z(_z) {}

    PBRT_CPU_GPU
    static Frame from_z(Vector3f z) {
        Vector3f x, y;
        z.coordinate_system(&x, &y);

        return Frame(x, y, z);
    }

    PBRT_CPU_GPU
    static Frame from_xz(Vector3f x, Vector3f z) {
        return Frame(x, z.cross(x), z);
    }

    PBRT_CPU_GPU
    Vector3f from_local(Vector3f v) const {
        return v.x * x + v.y * y + v.z * z;
    }

    PBRT_CPU_GPU
    Vector3f to_local(Vector3f v) const {
        return Vector3f(v.dot(x), v.dot(y), v.dot(z));
    }
};
