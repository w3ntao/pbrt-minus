#pragma once

#include "euclidean_space/squared_matrix.h"

class Transform {
    SquareMatrix<4> m;
    SquareMatrix<4> inv_m;

  public:
    PBRT_CPU_GPU Transform(const SquareMatrix<4> &_m, const SquareMatrix<4> &_inv_m)
        : m(_m), inv_m(_inv_m) {}

    PBRT_CPU_GPU static Transform translate(const Vector3f delta) {
        double data_m[4][4] = {
            {1, 0, 0, delta.x},
            {0, 1, 0, delta.y},
            {0, 0, 1, delta.z},
            {0, 0, 0, 1},
        };

        double data_inv_m[4][4] = {
            {1, 0, 0, -delta.x},
            {0, 1, 0, -delta.y},
            {0, 0, 1, -delta.z},
            {0, 0, 0, 1},
        };

        return Transform(SquareMatrix<4>(data_m), SquareMatrix<4>(data_inv_m));
    }

    template <typename T>
    PBRT_CPU_GPU Point3<T> operator()(const Point3<T> &p) const {
        T xp = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
        T yp = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
        T zp = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
        T wp = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];

        if (wp == 1) {
            return Point3<T>(xp, yp, zp);
        }

        return Point3<T>(xp, yp, zp) / wp;
    }
};
