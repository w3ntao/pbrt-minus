#pragma once

#include <stdexcept>

class Vector3 {
    public:
        double x, y, z;

        __host__ __device__ Vector3() : x(0), y(0), z(0){};

        __host__ __device__ Vector3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

        __host__ __device__ Vector3 operator+(const Vector3 &b) const {
            return Vector3(x + b.x, y + b.y, z + b.z);
        }

        __host__ __device__ Vector3 operator-(const Vector3 &b) const {
            return Vector3(x - b.x, y - b.y, z - b.z);
        }

        __host__ __device__ Vector3 operator-() const {
            return Vector3(-x, -y, -z);
        }

        __host__ __device__ Vector3 operator*(double factor) const {
            return Vector3(x * factor, y * factor, z * factor);
        }

        __host__ __device__ Vector3 operator/(double divisor) const {
            return Vector3(x / divisor, y / divisor, z / divisor);
        }

        __host__ __device__ inline double squared_length() const {
            return x * x + y * y + z * z;
        }

        __host__ __device__ inline double length() const {
            return std::sqrt(squared_length());
        }

        __host__ __device__ Vector3 normalize() const {
            return *this / length();
        }

        __host__ __device__ double &operator[](int index) {
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

        __host__ __device__ double operator[](int index) const {
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
};
__host__ __device__ Vector3 operator*(double factor, const Vector3 &v) {
    return v * factor;
}

__host__ __device__ double dot(const Vector3 &left, const Vector3 &right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

__host__ __device__ inline Vector3 cross(const Vector3 &left, const Vector3 &right) {
    return Vector3((left[1] * right[2] - left[2] * right[1]), (-(left[0] * right[2] - left[2] * right[0])),
                   (left[0] * right[1] - left[1] * right[0]));
}
