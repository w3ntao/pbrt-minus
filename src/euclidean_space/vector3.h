#pragma once

#include <stdexcept>

template <typename T>
class Vector3 {
    public:
        T x, y, z;

        PBRT_CPU_GPU Vector3() : x(0), y(0), z(0){};

        PBRT_CPU_GPU Vector3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

        PBRT_CPU_GPU T &operator[](int index) {
            switch (index) {
            case 0: {
                return x;
            }
            case 1: {
                return y;
            }
            case 2: {
                return z;
            }
            default: {
#if defined(__CUDA_ARCH__)
                asm("trap;");
#else
                throw std::runtime_error("Vector3: invalid index `" + std::to_string(index) + "`");
#endif
            }
            }
        }

        PBRT_CPU_GPU T operator[](int index) const {
            switch (index) {
            case 0: {
                return x;
            }
            case 1: {
                return y;
            }
            case 2: {
                return z;
            }
            default: {
#if defined(__CUDA_ARCH__)
                asm("trap;");
#else
                throw std::runtime_error("Vector3: invalid index `" + std::to_string(index) + "`");
#endif
            }
            }
        }

        PBRT_CPU_GPU Vector3 operator+(const Vector3 &b) const {
            return Vector3(x + b.x, y + b.y, z + b.z);
        }

        PBRT_CPU_GPU Vector3 operator-(const Vector3 &b) const {
            return Vector3(x - b.x, y - b.y, z - b.z);
        }

        PBRT_CPU_GPU Vector3 operator-() const {
            return Vector3(-x, -y, -z);
        }

        PBRT_CPU_GPU Vector3 operator*(T factor) const {
            return Vector3(x * factor, y * factor, z * factor);
        }

        PBRT_CPU_GPU Vector3 operator/(T divisor) const {
            return Vector3(x / divisor, y / divisor, z / divisor);
        }

        PBRT_CPU_GPU double squared_length() const {
            return x * x + y * y + z * z;
        }

        PBRT_CPU_GPU double length() const {
            return std::sqrt(squared_length());
        }

        PBRT_CPU_GPU Vector3 normalize() const {
            return *this / length();
        }

        PBRT_CPU_GPU Vector3 softmax() const {
            return Vector3(exp10f(x), exp10f(y), exp10f(z)).normalize();
        }

        PBRT_CPU_GPU T dot(const Vector3 &right) const {
            return x * right.x + y * right.y + z * right.z;
        }

        PBRT_CPU_GPU Vector3 cross(const Vector3 &right) const {
            return Vector3(y * right.z - z * right.y, -(x * right.z - z * right.x),
                           x * right.y - y * right.x);
        }
};

using Vector3f = Vector3<double>;

template <typename T>
PBRT_CPU_GPU Vector3<T> operator*(T factor, const Vector3<T> &v) {
    return v * factor;
}
