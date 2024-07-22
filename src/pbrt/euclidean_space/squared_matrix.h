#pragma once

#include <iomanip>
#include "pbrt/util/utility_math.h"
#include "pbrt/util/compensated_float.h"

template <uint N>
class SquareMatrix {
  public:
    PBRT_CPU_GPU SquareMatrix() {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                val[i][k] = i == k ? 1.0 : 0.0;
            }
        }
    }

    PBRT_CPU_GPU
    SquareMatrix(const FloatType data[N][N]) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                val[i][k] = data[i][k];
            }
        }
    }

    PBRT_CPU_GPU SquareMatrix(const SquareMatrix &matrix) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                val[i][k] = matrix.val[i][k];
            }
        }
    }

    PBRT_CPU_GPU static SquareMatrix zero() {
        FloatType data[N][N] = {0.0};
        return SquareMatrix(data);
    }

    PBRT_CPU_GPU static SquareMatrix nan() {
        FloatType data[N][N] = {NAN};
        return SquareMatrix(data);
    }

    PBRT_CPU_GPU static SquareMatrix diag(const FloatType data[N]) {
        FloatType m[N][N] = {0.0};
        for (int i = 0; i < N; i++) {
            m[i][i] = data[i];
        }

        return SquareMatrix(m);
    }

    PBRT_CPU_GPU static SquareMatrix identity() {
        FloatType data[N][N] = {0.0};
        for (int i = 0; i < N; i++) {
            data[i][i] = 1.0;
        }

        return {data};
    }

    PBRT_CPU_GPU void print() const {
        for (uint x = 0; x < N; x++) {
            printf("[ ");
            for (uint y = 0; y < N; y++) {
                printf("%f, ", val[x][y]);
            }
            printf("]\n");
        }
        printf("\n");
    }

    PBRT_CPU_GPU bool operator==(const SquareMatrix &matrix) const {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                if (val[i][k] != matrix.val[i][k]) {
                    return false;
                }
            }
        }

        return true;
    }

    PBRT_CPU_GPU bool operator!=(const SquareMatrix &matrix) const {
        return !(*this == matrix);
    }

    PBRT_CPU_GPU const FloatType *operator[](uint i) const {
        return val[i];
    }

    PBRT_CPU_GPU FloatType *operator[](uint i) {
        return val[i];
    }

    PBRT_CPU_GPU SquareMatrix operator*(const SquareMatrix &right) const {
        SquareMatrix r;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                r[i][j] = 0;
                for (int k = 0; k < N; ++k) {
                    r[i][j] = std::fma(val[i][k], right[k][j], r[i][j]);
                }
            }
        }
        return r;
    }

    PBRT_CPU_GPU
    FloatType determinant() const;

    PBRT_CPU_GPU
    SquareMatrix inverse() const;

    PBRT_CPU_GPU
    SquareMatrix transpose() const {
        SquareMatrix matrix;
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                matrix[y][x] = this->val[x][y];
            }
        }
        return matrix;
    }

    friend std::ostream &operator<<(std::ostream &stream, const SquareMatrix &t) {
        stream << "matrix " << N << "x" << N << " [\n";
        for (int i = 0; i < N; i++) {
            stream << "    ";
            for (int k = 0; k < N; k++) {
                stream << std::setprecision(6) << t[i][k] << ", ";
            }
            stream << "\n";
        }
        stream << "]\n";

        return stream;
    }

  private:
    FloatType val[N][N];
};
