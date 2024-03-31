#pragma once

#include <iomanip>
#include "pbrt/util/utility_math.h"
#include "pbrt/util/compensated_float.h"

template <int N>
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
    SquareMatrix(const double data[N][N]) {
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
        double data[N][N] = {0.0};
        return SquareMatrix(data);
    }

    PBRT_CPU_GPU static SquareMatrix diag(const double data[N]) {
        double m[N][N] = {0.0};
        for (int i = 0; i < N; i++) {
            m[i][i] = data[i];
        }

        return SquareMatrix(m);
    }

    PBRT_CPU_GPU static SquareMatrix identity() {
        double data[N][N] = {0.0};
        for (int i = 0; i < N; i++) {
            data[i][i] = 1.0;
        }

        return {data};
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

    PBRT_CPU_GPU double determinant() const {
        switch (N) {
        case 3: {
            const auto m = this->val;

            double minor12 = difference_of_products(m[1][1], m[2][2], m[1][2], m[2][1]);
            double minor02 = difference_of_products(m[1][0], m[2][2], m[1][2], m[2][0]);
            double minor01 = difference_of_products(m[1][0], m[2][1], m[1][1], m[2][0]);

            return std::fma(m[0][2], minor01,
                            difference_of_products(m[0][0], minor12, m[0][1], minor02));
        }
        case 4: {
            const auto m = this->val;

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

            return difference_of_products(s0, c5, s1, c4) +
                   difference_of_products(s2, c3, -s3, c2) + difference_of_products(s5, c0, s4, c1);
        }
        default: {
            printf("determinant() not implemented for SquareMatrix<%d>\n", N);

#if defined(__CUDA_ARCH__)
            asm("trap;");
#else
            throw std::runtime_error("SquareMatrix<N>::determinant() not implemented");
#endif
        }
        }
    }
    
    PBRT_CPU_GPU
    SquareMatrix inverse() const {
        switch (N) {
        case 3: {
            const double det = determinant();
            if (det == 0.0) {
                printf("can't inverse a singular matrix\n");
#if defined(__CUDA_ARCH__)
                asm("trap;");
#else
                throw std::runtime_error("singular matrix");
#endif
            }

            const double inv_det = 1.0 / det;

            const auto m = this->val;

            double r[3][3];
            r[0][0] = inv_det * difference_of_products(m[1][1], m[2][2], m[1][2], m[2][1]);
            r[1][0] = inv_det * difference_of_products(m[1][2], m[2][0], m[1][0], m[2][2]);
            r[2][0] = inv_det * difference_of_products(m[1][0], m[2][1], m[1][1], m[2][0]);
            r[0][1] = inv_det * difference_of_products(m[0][2], m[2][1], m[0][1], m[2][2]);
            r[1][1] = inv_det * difference_of_products(m[0][0], m[2][2], m[0][2], m[2][0]);
            r[2][1] = inv_det * difference_of_products(m[0][1], m[2][0], m[0][0], m[2][1]);
            r[0][2] = inv_det * difference_of_products(m[0][1], m[1][2], m[0][2], m[1][1]);
            r[1][2] = inv_det * difference_of_products(m[0][2], m[1][0], m[0][0], m[1][2]);
            r[2][2] = inv_det * difference_of_products(m[0][0], m[1][1], m[0][1], m[1][0]);

            SquareMatrix inverse_matrix;
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    inverse_matrix[x][y] = r[x][y];
                }
            }

            return inverse_matrix;
        }

        case 4: {
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

            const auto m = this->val;

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

            SquareMatrix inverse_matrix;
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    inverse_matrix[x][y] = inv[x][y];
                }
            }

            return inverse_matrix;
        }

        default: {
            printf("inverse() not implemented for SquareMatrix<%d>\n", N);

#if defined(__CUDA_ARCH__)
            asm("trap;");
#else
            throw std::runtime_error("SquareMatrix<N>::inverse() not implemented");
#endif
        }
        }
    }

    PBRT_CPU_GPU SquareMatrix transpose() const {
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
    double val[N][N];
};
