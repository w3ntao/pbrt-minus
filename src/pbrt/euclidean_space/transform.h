#pragma once

#include "pbrt/euclidean_space/squared_matrix.h"
#include "pbrt/euclidean_space/vector3fi.h"
#include "pbrt/euclidean_space/point3fi.h"
#include "pbrt/base/ray.h"

class Transform {
  public:
    SquareMatrix<4> m;
    SquareMatrix<4> inv_m;

    PBRT_CPU_GPU static Transform identity() {
        return {};
    }

    PBRT_CPU_GPU
    static Transform scale(double x, double y, double z) {
        double data_m[4][4] = {
            {x, 0, 0, 0},
            {0, y, 0, 0},
            {0, 0, z, 0},
            {0, 0, 0, 1},
        };

        double data_m_inv[4][4] = {
            {1.0 / x, 0, 0, 0},
            {0, 1.0 / y, 0, 0},
            {0, 0, 1.0 / z, 0},
            {0, 0, 0, 1},
        };

        return Transform(SquareMatrix(data_m), SquareMatrix(data_m_inv));
    }

    PBRT_CPU_GPU static Transform translate(double x, double y, double z) {
        const double data_m[4][4] = {
            {1, 0, 0, x},
            {0, 1, 0, y},
            {0, 0, 1, z},
            {0, 0, 0, 1},
        };

        const double data_inv_m[4][4] = {
            {1, 0, 0, -x},
            {0, 1, 0, -y},
            {0, 0, 1, -z},
            {0, 0, 0, 1},
        };

        return Transform(SquareMatrix(data_m), SquareMatrix(data_inv_m));
    }

    PBRT_CPU_GPU
    static Transform perspective(double fov, double z_near, double z_far) {
        double data_persp[4][4] = {
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, z_far / (z_far - z_near), -z_far * z_near / (z_far - z_near)},
            {0.0, 0.0, 1.0, 0.0},
        };
        auto persp = SquareMatrix(data_persp);

        auto inv_tan_ang = 1.0 / std::tan(degree_to_radian(fov) / 2.0);

        return scale(inv_tan_ang, inv_tan_ang, 1.0) * Transform(persp);
    }

    static Transform lookat(const Point3f &position, const Point3f &look, const Vector3f &up) {
        auto world_from_camera = SquareMatrix<4>::zero();
        world_from_camera[0][3] = position.x;
        world_from_camera[1][3] = position.y;
        world_from_camera[2][3] = position.z;
        world_from_camera[3][3] = 1.0;

        auto direction = (look - position).normalize();

        if (up.normalize().cross(direction).length() == 0.0) {
            throw std::invalid_argument(
                "LookAt: `up` vector and viewing direction are pointing in the same direction");
        }

        auto right = up.normalize().cross(direction).normalize();
        auto new_up = direction.cross(right);

        world_from_camera[0][0] = right.x;
        world_from_camera[1][0] = right.y;
        world_from_camera[2][0] = right.z;
        world_from_camera[3][0] = 0.0;
        world_from_camera[0][1] = new_up.x;
        world_from_camera[1][1] = new_up.y;
        world_from_camera[2][1] = new_up.z;
        world_from_camera[3][1] = 0.0;
        world_from_camera[0][2] = direction.x;
        world_from_camera[1][2] = direction.y;
        world_from_camera[2][2] = direction.z;
        world_from_camera[3][2] = 0.0;

        auto camera_from_world = world_from_camera.inverse();

        return Transform(camera_from_world, world_from_camera);
    }

    PBRT_CPU_GPU Transform() {
        m = SquareMatrix<4>::identity();
        inv_m = m;
    }

    PBRT_CPU_GPU
    Transform(const SquareMatrix<4> &_m) : m(_m), inv_m(m.inverse()) {}

    PBRT_CPU_GPU
    Transform(const SquareMatrix<4> &_m, const SquareMatrix<4> &_inv_m) : m(_m), inv_m(_inv_m) {}

    PBRT_CPU_GPU bool is_identity() const {
        if (m != inv_m) {
            return false;
        }

        for (int i = 0; i < 4; i++) {
            for (int k = 0; k < 4; k++) {
                if (i == k) {
                    if (m[i][k] != 1.0) {
                        return false;
                    }
                } else {
                    if (m[i][k] != 0.0) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    PBRT_CPU_GPU Transform inverse() const {
        return Transform(inv_m, m);
    }

    PBRT_CPU_GPU Transform operator*(const Transform &right) const {
        return Transform(m * right.m, right.inv_m * inv_m);
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

        if (wp == 1.0) {
            return Point3<T>(xp, yp, zp);
        }

        return Point3<T>(xp, yp, zp) / wp;
    }

    PBRT_CPU_GPU Vector3fi operator()(const Vector3fi &v) const {
        double x = double(v.x);
        double y = double(v.y);
        double z = double(v.z);
        Vector3f vOutError;
        if (v.IsExact()) {
            vOutError.x =
                gamma(3) * (std::abs(m[0][0] * x) + std::abs(m[0][1] * y) + std::abs(m[0][2] * z));
            vOutError.y =
                gamma(3) * (std::abs(m[1][0] * x) + std::abs(m[1][1] * y) + std::abs(m[1][2] * z));
            vOutError.z =
                gamma(3) * (std::abs(m[2][0] * x) + std::abs(m[2][1] * y) + std::abs(m[2][2] * z));
        } else {
            Vector3f vInError = v.Error();
            vOutError.x =
                (gamma(3) + 1) * (std::abs(m[0][0]) * vInError.x + std::abs(m[0][1]) * vInError.y +
                                  std::abs(m[0][2]) * vInError.z) +
                gamma(3) * (std::abs(m[0][0] * x) + std::abs(m[0][1] * y) + std::abs(m[0][2] * z));
            vOutError.y =
                (gamma(3) + 1) * (std::abs(m[1][0]) * vInError.x + std::abs(m[1][1]) * vInError.y +
                                  std::abs(m[1][2]) * vInError.z) +
                gamma(3) * (std::abs(m[1][0] * x) + std::abs(m[1][1] * y) + std::abs(m[1][2] * z));
            vOutError.z =
                (gamma(3) + 1) * (std::abs(m[2][0]) * vInError.x + std::abs(m[2][1]) * vInError.y +
                                  std::abs(m[2][2]) * vInError.z) +
                gamma(3) * (std::abs(m[2][0] * x) + std::abs(m[2][1] * y) + std::abs(m[2][2] * z));
        }

        double xp = m[0][0] * x + m[0][1] * y + m[0][2] * z;
        double yp = m[1][0] * x + m[1][1] * y + m[1][2] * z;
        double zp = m[2][0] * x + m[2][1] * y + m[2][2] * z;

        return Vector3fi(Vector3f(xp, yp, zp), vOutError);
    }

    template <typename T>
    PBRT_CPU_GPU Vector3<T> operator()(const Vector3<T> &v) const {
        return Vector3<T>(m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                          m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                          m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
    }

    PBRT_CPU_GPU Ray operator()(const Ray &r, double *tMax = nullptr) const {
        Point3fi o = (*this)(Point3fi(r.o));
        Vector3f d = (*this)(r.d);

        // Offset ray origin to edge of error bounds and compute _tMax_
        if (double lengthSquared = d.squared_length(); lengthSquared > 0) {
            double dt = d.abs().dot(o.error()) / lengthSquared;
            o += d * dt;
            if (tMax) {
                *tMax -= dt;
            }
        }

        return Ray(o.to_point3f(), d);
    }

    friend std::ostream &operator<<(std::ostream &stream, const Transform &transform) {
        stream << "Transform -- m: ";
        stream << transform.m;
        stream << "m_inv: " << transform.inv_m;

        return stream;
    }
};
