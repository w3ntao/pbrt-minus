#pragma once

#include "pbrt/util/macro.h"
#include "pbrt/util/accurate_arithmetic.h"
#include "pbrt/util/compensated_float.h"

template <int N>
class SquareMatrix {
  private:
    double val[N][N];

  public:
    PBRT_CPU_GPU SquareMatrix() {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                val[i][k] = i == k ? 1 : 0;
            }
        }
    }

    PBRT_CPU_GPU static SquareMatrix zero() {
        const double data[4][4] = {
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
        };

        return SquareMatrix(data);
    }

    PBRT_CPU_GPU static SquareMatrix identity() {
        const double data[4][4] = {
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 1},
        };

        return SquareMatrix(data);
    }

    PBRT_CPU_GPU SquareMatrix(const double data[N][N]) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                val[i][k] = data[i][k];
            }
        }
    }

    PBRT_CPU_GPU const double *operator[](int i) const {
        return val[i];
    }

    PBRT_CPU_GPU double *operator[](int i) {
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

    PBRT_CPU_GPU SquareMatrix inverse() const;

    friend std::ostream &operator<<(std::ostream &stream, const SquareMatrix &t) {
        stream << "matrix " << N << "x" << N << " [\n";
        for (int i = 0; i < N; i++) {
            stream << "    ";
            for (int k = 0; k < N; k++) {
                stream << t[i][k] << ", ";
            }
            stream << "\n";
        }
        stream << "]\n";

        return stream;
    }
};

template <>
PBRT_CPU_GPU inline SquareMatrix<4> SquareMatrix<4>::inverse() const {
    /*
    Via: https://github.com/google/ion/blob/master/ion/math/matrixutils.cc,
    (c) Google, Apache license.

    For 4x4 do not compute the adjugate as the transpose of the cofactor
    matrix, because this results in extra work. Several calculations can be
    shared across the sub-determinants.
    //
    This approach is explained in David Eberly's Geometric Tools book,
    excerpted here:
      http://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
    */

    auto m = this->val;

    double s0 = difference_of_products(m[0][0], m[1][1], m[1][0], m[0][1]);
    double s1 = difference_of_products(m[0][0], m[1][2], m[1][0], m[0][2]);
    double s2 = difference_of_products(m[0][0], m[1][3], m[1][0], m[0][3]);

    double s3 = difference_of_products(m[0][1], m[1][2], m[1][1], m[0][2]);
    double s4 = difference_of_products(m[0][1], m[1][3], m[1][1], m[0][3]);
    double s5 = difference_of_products(m[0][2], m[1][3], m[1][2], m[0][3]);

    double c0 = difference_of_products(m[2][0], m[3][1], m[3][0], m[2][1]);
    double c1 = difference_of_products(m[2][0], m[3][2], m[3][0], m[2][2]);
    double c2 = difference_of_products(m[2][0], m[3][3], m[3][0], m[2][3]);

    double c3 = difference_of_products(m[2][1], m[3][2], m[3][1], m[2][2]);
    double c4 = difference_of_products(m[2][1], m[3][3], m[3][1], m[2][3]);
    double c5 = difference_of_products(m[2][2], m[3][3], m[3][2], m[2][3]);

    double determinant = inner_product(s0, c5, -s1, c4, s2, c3, s3, c2, s5, c0, -s4, c1);
    if (determinant == 0) {
        printf("can't inverse a singular matrix\n");
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::runtime_error("singular matrix");
#endif
    }
    double s = 1.0 / determinant;

    double inv[4][4] = {{
                            s * inner_product(m[1][1], c5, m[1][3], c3, -m[1][2], c4),
                            s * inner_product(-m[0][1], c5, m[0][2], c4, -m[0][3], c3),
                            s * inner_product(m[3][1], s5, m[3][3], s3, -m[3][2], s4),
                            s * inner_product(-m[2][1], s5, m[2][2], s4, -m[2][3], s3),
                        },
                        {
                            s * inner_product(-m[1][0], c5, m[1][2], c2, -m[1][3], c1),
                            s * inner_product(m[0][0], c5, m[0][3], c1, -m[0][2], c2),
                            s * inner_product(-m[3][0], s5, m[3][2], s2, -m[3][3], s1),
                            s * inner_product(m[2][0], s5, m[2][3], s1, -m[2][2], s2),
                        },
                        {
                            s * inner_product(m[1][0], c4, m[1][3], c0, -m[1][1], c2),
                            s * inner_product(-m[0][0], c4, m[0][1], c2, -m[0][3], c0),
                            s * inner_product(m[3][0], s4, m[3][3], s0, -m[3][1], s2),
                            s * inner_product(-m[2][0], s4, m[2][1], s2, -m[2][3], s0),
                        },
                        {
                            s * inner_product(-m[1][0], c3, m[1][1], c1, -m[1][2], c0),
                            s * inner_product(m[0][0], c3, m[0][2], c0, -m[0][1], c1),
                            s * inner_product(-m[3][0], s3, m[3][1], s1, -m[3][2], s0),
                            s * inner_product(m[2][0], s3, m[2][2], s0, -m[2][1], s1),
                        }};

    return SquareMatrix(inv);
}
