#pragma once

#include "pbrt/euclidean_space/squared_matrix.h"
#include "pbrt/euclidean_space/point3.h"
#include "pbrt/euclidean_space/vector3.h"

class Transform {

  public:
    SquareMatrix<4> m;
    SquareMatrix<4> inv_m;

    PBRT_CPU_GPU static Transform identity() {
        return {};
    }

    PBRT_CPU_GPU static Transform translate(const Vector3f &delta) {
        const double data_m[4][4] = {
            {1, 0, 0, delta.x},
            {0, 1, 0, delta.y},
            {0, 0, 1, delta.z},
            {0, 0, 0, 1},
        };

        const double data_inv_m[4][4] = {
            {1, 0, 0, -delta.x},
            {0, 1, 0, -delta.y},
            {0, 0, 1, -delta.z},
            {0, 0, 0, 1},
        };

        return Transform(SquareMatrix(data_m), SquareMatrix(data_inv_m));
    }
    static Transform lookat(const Point3f &pos, const Point3f &look, const Vector3f &up) {
        auto world_from_camera = SquareMatrix<4>::zero();
        world_from_camera[0][3] = pos.x;
        world_from_camera[1][3] = pos.y;
        world_from_camera[2][3] = pos.z;
        world_from_camera[3][3] = 1.0;

        auto dir = (look - pos).normalize();

        if (up.normalize().cross(dir).length() == 0.0) {
            throw std::invalid_argument(
                "LookAt: `up` vector and viewing direction are pointing in the same direction");
        }

        auto right = up.normalize().cross(dir).normalize();
        auto new_up = dir.cross(right);

        world_from_camera[0][0] = right.x;
        world_from_camera[1][0] = right.y;
        world_from_camera[2][0] = right.z;
        world_from_camera[3][0] = 0.0;
        world_from_camera[0][1] = new_up.x;
        world_from_camera[1][1] = new_up.y;
        world_from_camera[2][1] = new_up.z;
        world_from_camera[3][1] = 0.0;
        world_from_camera[0][2] = dir.x;
        world_from_camera[1][2] = dir.y;
        world_from_camera[2][2] = dir.z;
        world_from_camera[3][2] = 0.0;

        auto camera_from_world = world_from_camera.inverse();

        return Transform(camera_from_world, world_from_camera);
    }

    PBRT_CPU_GPU Transform() {
        m = SquareMatrix<4>::identity();
        inv_m = m;
    }

    PBRT_CPU_GPU Transform(const SquareMatrix<4> &_m, const SquareMatrix<4> &_inv_m)
        : m(_m), inv_m(_inv_m) {}

    PBRT_CPU_GPU Transform operator*(const Transform &right) const {
        return Transform(m * right.m, inv_m * right.inv_m);
    }

    PBRT_CPU_GPU void operator*=(const Transform &right) {
        *this = this->operator*(right);
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

    friend std::ostream &operator<<(std::ostream &stream, const Transform &transform) {
        stream << "Transform -- m: ";
        stream << transform.m;
        stream << "m_inv: " << transform.inv_m;

        return stream;
    }
};
