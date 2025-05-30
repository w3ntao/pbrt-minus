#pragma once

#include <pbrt/euclidean_space/bounds3.h>
#include <pbrt/euclidean_space/normal3f.h>
#include <pbrt/euclidean_space/point3fi.h>
#include <pbrt/euclidean_space/squared_matrix.h>
#include <pbrt/euclidean_space/vector3fi.h>

class Frame;
class Ray;
class SurfaceInteraction;

class Transform {
  private:
    SquareMatrix<4> m;
    SquareMatrix<4> inv_m;

  public:
    PBRT_CPU_GPU Transform() {
        m = SquareMatrix<4>::identity();
        inv_m = m;
    }

    PBRT_CPU_GPU
    explicit Transform(const Real values[16]) {
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                auto idx = y * 4 + x;
                m[y][x] = values[idx];
            }
        }

        inv_m = m.inverse();
    }

    PBRT_CPU_GPU
    explicit Transform(const Real values[4][4]) {
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                m[y][x] = values[y][x];
            }
        }

        inv_m = m.inverse();
    }

    PBRT_CPU_GPU
    explicit Transform(const Frame &frame);

    PBRT_CPU_GPU
    Transform(const SquareMatrix<4> &_m) : m(_m), inv_m(m.inverse()) {}

    PBRT_CPU_GPU
    Transform(const SquareMatrix<4> &_m, const SquareMatrix<4> &_inv_m) : m(_m), inv_m(_inv_m) {}

    PBRT_CPU_GPU static Transform identity() {
        return {};
    }

    PBRT_CPU_GPU
    static Transform rotate(Real angle, Real x, Real y, Real z) {
        auto sin_theta = std::sin(degree_to_radian(angle));
        auto cos_theta = std::cos(degree_to_radian(angle));
        auto axis = Vector3f(x, y, z).normalize();

        auto m = SquareMatrix<4>::identity();
        // Compute rotation of first basis vector
        m[0][0] = axis.x * axis.x + (1.0 - axis.x * axis.x) * cos_theta;
        m[0][1] = axis.x * axis.y * (1.0 - cos_theta) - axis.z * sin_theta;
        m[0][2] = axis.x * axis.z * (1.0 - cos_theta) + axis.y * sin_theta;
        m[0][3] = 0.0;

        // Compute rotations of second and third basis vectors
        m[1][0] = axis.x * axis.y * (1.0 - cos_theta) + axis.z * sin_theta;
        m[1][1] = axis.y * axis.y + (1.0 - axis.y * axis.y) * cos_theta;
        m[1][2] = axis.y * axis.z * (1.0 - cos_theta) - axis.x * sin_theta;
        m[1][3] = 0.0;

        m[2][0] = axis.x * axis.z * (1.0 - cos_theta) - axis.y * sin_theta;
        m[2][1] = axis.y * axis.z * (1.0 - cos_theta) + axis.x * sin_theta;
        m[2][2] = axis.z * axis.z + (1.0 - axis.z * axis.z) * cos_theta;
        m[2][3] = 0.0;

        Transform t;
        t.m = m;
        t.inv_m = m.transpose();
        return t;
    }

    PBRT_CPU_GPU
    static Transform scale(Real x, Real y, Real z) {
        Real data_m[4][4] = {
            {x, 0, 0, 0},
            {0, y, 0, 0},
            {0, 0, z, 0},
            {0, 0, 0, 1},
        };

        Real data_m_inv[4][4] = {
            {Real(1.0) / x, 0, 0, 0},
            {0, Real(1.0) / y, 0, 0},
            {0, 0, Real(1.0) / z, 0},
            {0, 0, 0, 1},
        };

        return Transform(SquareMatrix(data_m), SquareMatrix(data_m_inv));
    }

    PBRT_CPU_GPU static Transform translate(Real x, Real y, Real z) {
        const Real data_m[4][4] = {
            {1, 0, 0, x},
            {0, 1, 0, y},
            {0, 0, 1, z},
            {0, 0, 0, 1},
        };

        const Real data_inv_m[4][4] = {
            {1, 0, 0, -x},
            {0, 1, 0, -y},
            {0, 0, 1, -z},
            {0, 0, 0, 1},
        };

        return Transform(SquareMatrix(data_m), SquareMatrix(data_inv_m));
    }

    PBRT_CPU_GPU
    static Transform perspective(Real fov, Real z_near, Real z_far) {
        Real data_persp[4][4] = {
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, z_far / (z_far - z_near), -z_far * z_near / (z_far - z_near)},
            {0.0, 0.0, 1.0, 0.0},
        };
        auto persp = SquareMatrix(data_persp);

        auto inv_tan_ang = 1.0 / std::tan(degree_to_radian(fov) / 2.0);

        return scale(inv_tan_ang, inv_tan_ang, 1.0) * Transform(persp);
    }

    PBRT_CPU_GPU
    static Transform rotate_from_to(const Vector3f from, const Vector3f to) {
        // Compute intermediate vector for vector reflection
        const Real threshold = 0.72;
        Vector3f refl;
        if (std::abs(from.x) < threshold && std::abs(to.x) < threshold) {
            refl = Vector3f(1, 0, 0);
        } else if (std::abs(from.y) < threshold && std::abs(to.y) < threshold) {
            refl = Vector3f(0, 1, 0);
        } else {
            refl = Vector3f(0, 0, 1);
        }

        // Initialize matrix _r_ for rotation
        Vector3f u = refl - from;
        Vector3f v = refl - to;

        SquareMatrix<4> r;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // Initialize matrix element _r[i][j]_
                r[i][j] = ((i == j) ? 1.0 : 0.0) - 2.0 / u.dot(u) * u[i] * u[j] -
                          2 / v.dot(v) * v[i] * v[j] +
                          4 * u.dot(v) / (u.dot(u) * v.dot(v)) * v[i] * u[j];
            }
        }

        return Transform(r, r.transpose());
    }

    static Transform lookat(const Point3f &position, const Point3f &look, const Vector3f &up) {
        auto world_from_camera = SquareMatrix<4>::zero();
        world_from_camera[0][3] = position.x;
        world_from_camera[1][3] = position.y;
        world_from_camera[2][3] = position.z;
        world_from_camera[3][3] = 1.0;

        auto direction = (look - position).normalize();

        if (up.normalize().cross(direction).length() == 0.0) {
            printf("%s(): `up` vector and viewing direction are pointing in the same direction\n\n",
                   __func__);
            REPORT_FATAL_ERROR();
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

    bool swaps_handedness() const {
        SquareMatrix<3> s;
        for (int x = 0; x < 3; ++x) {
            for (int y = 0; y < 3; ++y) {
                s[x][y] = m[x][y];
            }
        }

        return s.determinant() < 0;
    }

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
        Real x = Real(v.x);
        Real y = Real(v.y);
        Real z = Real(v.z);
        Vector3f vOutError;
        if (v.is_exact()) {
            vOutError.x =
                gamma(3) * (std::abs(m[0][0] * x) + std::abs(m[0][1] * y) + std::abs(m[0][2] * z));
            vOutError.y =
                gamma(3) * (std::abs(m[1][0] * x) + std::abs(m[1][1] * y) + std::abs(m[1][2] * z));
            vOutError.z =
                gamma(3) * (std::abs(m[2][0] * x) + std::abs(m[2][1] * y) + std::abs(m[2][2] * z));
        } else {
            Vector3f vInError = v.error();
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

        Real xp = m[0][0] * x + m[0][1] * y + m[0][2] * z;
        Real yp = m[1][0] * x + m[1][1] * y + m[1][2] * z;
        Real zp = m[2][0] * x + m[2][1] * y + m[2][2] * z;

        return Vector3fi(Vector3f(xp, yp, zp), vOutError);
    }

    template <typename T>
    PBRT_CPU_GPU Vector3<T> operator()(const Vector3<T> &v) const {
        return Vector3<T>(m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                          m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                          m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
    }

    PBRT_CPU_GPU Normal3f operator()(const Normal3f &n) const {
        Real x = n.x;
        Real y = n.y;
        Real z = n.z;

        return Normal3f(inv_m[0][0] * x + inv_m[1][0] * y + inv_m[2][0] * z,
                        inv_m[0][1] * x + inv_m[1][1] * y + inv_m[2][1] * z,
                        inv_m[0][2] * x + inv_m[1][2] * y + inv_m[2][2] * z);
    }

    PBRT_CPU_GPU
    Ray operator()(const Ray &r, Real *tMax = nullptr) const;

    PBRT_CPU_GPU
    Bounds3f operator()(const Bounds3f &bounds) const {
        // a smarter way to transform bounds:
        // takes roughly 2 transforms instead of 8
        // https://stackoverflow.com/a/58630206

        auto transformed_bounds = Bounds3f ::empty();
        for (int idx = 0; idx < 3; ++idx) {
            transformed_bounds.p_min[idx] = m[idx][3];
        }
        transformed_bounds.p_max = transformed_bounds.p_min;

        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 3; ++k) {
                auto a = m[i][k] * bounds.p_min[k];
                auto b = m[i][k] * bounds.p_max[k];

                auto min_val = std::min(a, b);
                auto max_val = std::max(a, b);

                transformed_bounds.p_min[i] += min_val;
                transformed_bounds.p_max[i] += max_val;
            }
        }

        return transformed_bounds;
    }

    PBRT_CPU_GPU
    SurfaceInteraction operator()(const SurfaceInteraction &si) const;

    template <typename T>
    PBRT_CPU_GPU Vector3<T> apply_inverse(const Vector3<T> v) const {
        T x = v.x;
        T y = v.y;
        T z = v.z;

        return Vector3<T>(inv_m[0][0] * x + inv_m[0][1] * y + inv_m[0][2] * z,
                          inv_m[1][0] * x + inv_m[1][1] * y + inv_m[1][2] * z,
                          inv_m[2][0] * x + inv_m[2][1] * y + inv_m[2][2] * z);
    }

    template <typename T>
    PBRT_CPU_GPU Point3<T> apply_inverse(const Point3<T> p) const {
        T x = p.x, y = p.y, z = p.z;

        T xp = (inv_m[0][0] * x + inv_m[0][1] * y) + (inv_m[0][2] * z + inv_m[0][3]);
        T yp = (inv_m[1][0] * x + inv_m[1][1] * y) + (inv_m[1][2] * z + inv_m[1][3]);
        T zp = (inv_m[2][0] * x + inv_m[2][1] * y) + (inv_m[2][2] * z + inv_m[2][3]);
        T wp = (inv_m[3][0] * x + inv_m[3][1] * y) + (inv_m[3][2] * z + inv_m[3][3]);

        return wp == 1 ? Point3<T>(xp, yp, zp) : Point3<T>(xp, yp, zp) / wp;
    }

    PBRT_CPU_GPU
    Point3fi apply_inverse(const Point3fi &p) const {
        auto x = p.x.midpoint();
        auto y = p.y.midpoint();
        auto z = p.z.midpoint();

        // Compute transformed coordinates from point _pt_
        Real xp = (inv_m[0][0] * x + inv_m[0][1] * y) + (inv_m[0][2] * z + inv_m[0][3]);
        Real yp = (inv_m[1][0] * x + inv_m[1][1] * y) + (inv_m[1][2] * z + inv_m[1][3]);
        Real zp = (inv_m[2][0] * x + inv_m[2][1] * y) + (inv_m[2][2] * z + inv_m[2][3]);
        Real wp = (inv_m[3][0] * x + inv_m[3][1] * y) + (inv_m[3][2] * z + inv_m[3][3]);

        // Compute absolute error for transformed point
        Vector3f pOutError;
        if (p.is_exact()) {
            pOutError.x = gamma(3) * (std::abs(inv_m[0][0] * x) + std::abs(inv_m[0][1] * y) +
                                      std::abs(inv_m[0][2] * z));
            pOutError.y = gamma(3) * (std::abs(inv_m[1][0] * x) + std::abs(inv_m[1][1] * y) +
                                      std::abs(inv_m[1][2] * z));
            pOutError.z = gamma(3) * (std::abs(inv_m[2][0] * x) + std::abs(inv_m[2][1] * y) +
                                      std::abs(inv_m[2][2] * z));
        } else {
            Vector3f pInError = p.error();
            pOutError.x = (gamma(3) + 1) * (std::abs(inv_m[0][0]) * pInError.x +
                                            std::abs(inv_m[0][1]) * pInError.y +
                                            std::abs(inv_m[0][2]) * pInError.z) +
                          gamma(3) * (std::abs(inv_m[0][0] * x) + std::abs(inv_m[0][1] * y) +
                                      std::abs(inv_m[0][2] * z) + std::abs(inv_m[0][3]));
            pOutError.y = (gamma(3) + 1) * (std::abs(inv_m[1][0]) * pInError.x +
                                            std::abs(inv_m[1][1]) * pInError.y +
                                            std::abs(inv_m[1][2]) * pInError.z) +
                          gamma(3) * (std::abs(inv_m[1][0] * x) + std::abs(inv_m[1][1] * y) +
                                      std::abs(inv_m[1][2] * z) + std::abs(inv_m[1][3]));
            pOutError.z = (gamma(3) + 1) * (std::abs(inv_m[2][0]) * pInError.x +
                                            std::abs(inv_m[2][1]) * pInError.y +
                                            std::abs(inv_m[2][2]) * pInError.z) +
                          gamma(3) * (std::abs(inv_m[2][0] * x) + std::abs(inv_m[2][1] * y) +
                                      std::abs(inv_m[2][2] * z) + std::abs(inv_m[2][3]));
        }

        if (wp == 1) {
            return Point3fi(Point3f(xp, yp, zp), pOutError);
        }

        return Point3fi(Point3f(xp, yp, zp), pOutError) / wp;
    }

    PBRT_CPU_GPU
    Ray apply_inverse(const Ray &r, Real *tMax) const;

    friend std::ostream &operator<<(std::ostream &stream, const Transform &transform) {
        stream << "Transform -- m: ";
        stream << transform.m;
        stream << "m_inv: " << transform.inv_m;

        return stream;
    }
};
